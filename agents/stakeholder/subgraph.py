"""v0.3: 원문 검증 → 관점 결과 / idempotent evidence delta."""
import json
import re
from copy import deepcopy
from datetime import date

from langgraph.graph import END, START, StateGraph

from .evidence import merge_evidence
from .models import Extraction, StakeholderState
from .web import digest, normalize

GROUPS = ('competitor', 'adopter', 'investor')
GROUP_LABELS = {'competitor': '경쟁 기술 진영', 'adopter': '도입 기업·개발자', 'investor': '투자·산업 관계자'}


GROUP_QUERIES = {
    'competitor': {'r1': '{name} competing approaches alternatives public stance response',
                   'r1c': '{name} versus alternatives comparison drawbacks criticism'},
    'adopter': {'r1': '{name} deployment experience production usage review adoption',
                'r1c': '{name} issues limitations problems GitHub issue discussion migration cost'},
    'investor': {'r1': '{name} investment analyst market outlook industry report',
                 'r1c': '{name} analyst risk skepticism concerns adoption challenges'},
}


def default_request(as_of_date=None):
    return {
        'sw': {'name': 'DeepSeek-V2 Multi-head Latent Attention (MLA)', 'selection_reason': '모델 구조의 KV 압축',
               'seed_urls': ['https://arxiv.org/abs/2405.04434']},
        'hw': {'name': 'ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories',
               'selection_reason': 'CXL 기반 계층 확장', 'seed_urls': ['https://arxiv.org/abs/2606.12556']},
        'domain': 'datacenter', 'as_of_date': as_of_date or date.today().isoformat(), 'language': 'ko',
        'max_search_rounds': 2, 'max_revision_rounds': 1, 'max_queries': 12,
    }


def unique(items):
    return list(dict.fromkeys(items))


def pages_from(batches):
    return {url: page for batch in batches for url, page in batch.get('pages', {}).items()}


MAX_QUOTE_WORDS = 20
MAX_QUOTE_CHARS = 240


def shorten_quote(quote):
    """공백을 정규화한 인용의 앞부분을 단어 단위로 자른다 (원문의 부분 문자열이 유지된다)."""
    words = normalize(quote).split()[:MAX_QUOTE_WORDS]
    while len(' '.join(words)) > MAX_QUOTE_CHARS and len(words) > 1:
        words.pop()
    return ' '.join(words)[:MAX_QUOTE_CHARS]


def url_of(reason):
    """기각 사유 문자열 '사유: URL' 에서 URL 만 꺼낸다."""
    return reason.split(': ', 1)[1] if ': ' in reason else ''


def problem_pairs(batches, rejected):
    """기각된 URL 을 찾아 온 질의의 (기술, 그룹). 이 쌍만 '검증 미완료'로 본다."""
    urls = {url_of(r) for r in rejected}
    return {(log['technology_id'], log['group']) for b in batches for log in b.get('search_logs', [])
            if urls & set(log.get('urls', []))}


def validate_observations(extraction, batches, as_of):
    pages = pages_from(batches)
    accepted, rejected, seen = [], [], set()
    for item in extraction.observations:
        page = pages.get(item.source_url)
        if not page or page['status'] != 'ok':
            rejected.append(f'원문 접근 미확인: {item.source_url}')
            continue
        content_hash = digest(json.dumps(page['blocks'], ensure_ascii=False, sort_keys=True))
        block = next((b for b in page['blocks'] if b['locator'] == item.page_or_locator), None)
        if page['content_hash'] != content_hash or not block or not item.quote or normalize(item.quote) not in normalize(block['text']):
            rejected.append(f'원문 hash/locator/quote 불일치: {item.source_url}')
            continue
        if len(item.quote.split()) > MAX_QUOTE_WORDS or len(item.quote) > MAX_QUOTE_CHARS:
            # 원문의 연속 구간이므로 앞부분만 쓰면 그대로 검증을 통과한다. 통째로 버리지 않고 줄인다.
            item = item.model_copy(update={'quote': shorten_quote(item.quote),
                                           'uncertainty': (item.uncertainty + ' 인용을 20단어 이내로 줄임.').strip()})
        if item.domain_relevance != 'datacenter' or not item.speaker.strip() or not item.statement.strip():
            rejected.append(f'도메인·발언 정보 미확인: {item.source_url}')
            continue
        if ((item.stance == 'positive' and item.evidence_stance != 'support')
                or (item.stance == 'negative' and item.evidence_stance != 'counter')
                or (item.stance in ('neutral', 'unknown') and item.evidence_stance != 'neutral')
                or (item.stance == 'conditional' and item.evidence_stance == 'counter' and not item.conditions)):
            rejected.append(f'입장·근거 방향 불일치: {item.source_url}')
            continue
        # 원문에 없는 수치/단위를 요약에 추가하면 제거. 인과·비교 기준의 타당성은 별도 Judge/검토 대상.
        tokens = re.findall(r'\d+(?:[.,]\d+)*(?:%|×)?|\b(?:GB/s|MB/s|GB|TB|ms|TTFT|TPOT)\b', item.statement)
        if any(token not in block['text'] for token in tokens):
            rejected.append(f'수치·단위가 인용 block에 없음: {item.source_url}')
            continue
        if item.published_date:
            try:
                stamp = date.fromisoformat(item.published_date)
                if stamp > date.fromisoformat(as_of):
                    raise ValueError()
            except ValueError:
                rejected.append(f'날짜 오류 또는 기준일 이후: {item.source_url}')
                continue
            metadata_text = json.dumps(page.get('metadata', {}), ensure_ascii=False)
            original_text = '\n'.join(b['text'] for b in page['blocks'])
            if item.published_date not in metadata_text and item.published_date not in original_text:
                item = item.model_copy(update={'published_date': None, 'uncertainty': item.uncertainty + ' 발행일 원문 확인 불가.'})
        key = (item.technology_id, item.group, item.speaker, item.source_url, item.page_or_locator, item.statement, item.evidence_stance)
        if key not in seen:
            seen.add(key)
            accepted.append(item.model_copy(update={'quote': normalize(item.quote)}))
    return accepted, unique(rejected)


def search_outcomes(observations, batches, problems=()):
    """기술×그룹×근거방향 별 found / not_found / blocked / unsearched.

    problems: 검증 미완료 쌍 {(기술, 그룹)}. 이 쌍만 not_found 대신 blocked 가 된다 (다른 쌍은 영향받지 않는다).
    """
    logs = [log for b in batches for log in b.get('search_logs', [])]
    pages = pages_from(batches)

    def completed(log):
        if log['status'] in ('ok', 'no_results'):
            return True
        # 일부 원문만 접근 실패한 경우: 읽은 원문이 하나라도 있으면 검색은 끝난 것으로 본다.
        return log['status'] == 'access_incomplete' and any(pages.get(u, {}).get('status') == 'ok' for u in log.get('urls', []))

    outcomes = []
    for side in ('sw', 'hw'):
        for group in GROUPS:
            scoped = [q for q in logs if q['technology_id'] == side and q['group'] == group]
            successful = [q for q in scoped if completed(q)]
            for stance in ('support', 'counter', 'neutral'):
                found = any(o.technology_id == side and o.group == group and o.target_scope == 'selected_technology'
                            and o.evidence_stance == stance for o in observations)
                clean = successful and (side, group) not in problems
                status = 'found' if found else 'not_found' if clean else 'blocked' if scoped else 'unsearched'
                outcomes.append({'technology_id': side, 'group': group, 'stance': stance, 'status': status,
                                 'query_ids': [q['id'] for q in scoped],
                                 'scope_note': '한정된 질의·기간·최대 5개 원문 내 결과이며, 의견의 부재를 증명하지 않음'})
    return outcomes


def make_findings(state):
    extraction = Extraction.model_validate(state['extraction'])
    observations, rejected = validate_observations(extraction, state['batches'], state['request']['as_of_date'])
    pages = pages_from(state['batches'])
    gaps = unique(state['gaps'] + state['rejected'] + rejected)
    claims, positions, old_evidence, pool = [], [], [], {}
    for item in observations:
        page = pages[item.source_url]
        url = page['final_url']
        eid = 'ev:' + digest(json.dumps([url, item.page_or_locator, normalize(item.quote)], ensure_ascii=False))
        cid = 'stakeholder:claim:' + digest(json.dumps([item.technology_id, item.group, item.speaker, item.statement, eid], ensure_ascii=False))
        binding = {'claim_id': cid, 'perspective': 'stakeholder', 'stance': item.evidence_stance}
        ev = {'id': eid, 'claim_id': cid, 'doc_id': None, 'title': item.source_title,
              'author_or_org': item.speaker, 'source_type': item.source_type,
              'primary_or_secondary': item.primary_or_secondary, 'direct_or_proxy': item.direct_or_proxy,
              'evidence_level': item.evidence_level,
              'url': url, 'published_at': item.published_date, 'accessed_at': page['accessed_at'],
              'page_or_locator': item.page_or_locator, 'quote': item.quote, 'stance': item.evidence_stance,
              'perspective': 'stakeholder', 'content_hash': page['content_hash'],
              'claim_ids': [cid], 'perspectives': ['stakeholder'], 'bindings': [binding], 'snapshot_path': page.get('snapshot_path')}
        pool = merge_evidence(pool, {eid: ev})
        if not item.published_date:
            gaps.append('원문 발행일 미확인: ' + url)
        claims.append({'claim_id': cid, 'technology_ids': [item.technology_id], 'topic': item.group,
                       'statement': item.statement, 'basis': 'direct_evidence', 'evidence_ids': [eid],
                       'conditions': item.conditions, 'uncertainty': item.uncertainty})
        positions.append({'technology_id': item.technology_id, 'group': item.group, 'target_name': item.target_name,
                          'target_scope': item.target_scope, 'speaker': item.speaker, 'affiliation': item.affiliation,
                          'stance': item.stance, 'evidence_stance': item.evidence_stance, 'claim_ids': [cid], 'bias_notes': item.bias_notes})
    for ev in pool.values():
        old_evidence.append({'evidence_id': ev['id'], 'title': ev['title'], 'url': ev['url'],
                             'published_date': ev['published_at'], 'excerpt': ev['quote']})
    outcomes = search_outcomes(observations, state['batches'], problem_pairs(state['batches'], [*state['rejected'], *rejected]))
    unresolved = [o for o in outcomes if o['status'] in ('blocked', 'unsearched')]
    gaps.extend(f"{o['technology_id']}/{o['group']}/{o['stance']}: {o['status']}" for o in unresolved)
    status = 'failed' if not observations and state['errors'] else 'partial' if gaps or state['errors'] else 'complete'
    return {'claims': claims, 'positions': positions, 'evidence': old_evidence, 'evidence_store': pool,
            'search_outcomes': outcomes, 'completion': {'status': status, 'search_rounds_used': state['rounds'],
                'revision_rounds_used': state['revision_rounds'], 'gaps': unique(gaps), 'errors': state['errors']}}


def build_stakeholder_graph(backend):
    def plan(state):
        request = state['request']
        focus = ' '.join(request.get('focus_queries') or [])[:200]
        only = {tuple(pair) for pair in request.get('only_pairs') or []}
        queries = []
        for side in ('sw', 'hw'):
            for group in GROUPS:
                if only and (side, group) not in only:
                    continue
                # 쌍마다 지지 쪽(r1)과 우려 쪽(r1c) 질의를 따로 보낸다. 한 질의로는 반대 의견이 거의 안 잡힌다.
                for suffix, template in GROUP_QUERIES[group].items():
                    text = template.format(name=request[side]['name']) + ' data center LLM inference'
                    queries.append({'id': f'{side}:{group}:{suffix}', 'technology_id': side, 'group': group,
                                    'query': text + (f' {focus}' if focus else '')})
        return {'queries': queries[:request['max_queries']]}

    def search(state):
        try:
            batch = backend.search(state['request'], state['queries'], state['technical_findings'])
            return {'batches': [*state['batches'], batch], 'rounds': state['rounds'] + 1,
                    'errors': state['errors'] + batch.get('errors', [])}
        except Exception as exc:
            return {'rounds': state['rounds'] + 1, 'errors': state['errors'] + [f'검색 실패 ({type(exc).__name__})']}

    def extract(state):
        if not state['batches']:
            return {'extraction': {'observations': [], 'gaps': []}}
        try:
            result = backend.extract(state['request'], state['batches'], state['rejected'])
            return {'extraction': result.model_dump()}
        except Exception as exc:
            return {'errors': state['errors'] + [f'구조화 실패 ({type(exc).__name__})']}

    def review(state):
        extraction = Extraction.model_validate(state['extraction'])
        accepted, rejected = validate_observations(extraction, state['batches'], state['request']['as_of_date'])
        outcomes = search_outcomes(accepted, state['batches'], problem_pairs(state['batches'], rejected))
        gaps = extraction.gaps + [f"{o['technology_id']}/{o['group']}: {o['status']}" for o in outcomes if o['status'] in ('blocked', 'unsearched')]
        return {'rejected': rejected, 'gaps': unique(gaps), 'extraction': {'observations': [o.model_dump() for o in accepted], 'gaps': extraction.gaps}}

    def route(state):
        if state['errors']:
            return 'finish'
        if state['rejected'] and state['revision_rounds'] < state['request']['max_revision_rounds']:
            return 'revise'
        used = sum(len(b.get('search_logs', [])) for b in state['batches'])
        if state['gaps'] and state['rounds'] < state['request']['max_search_rounds'] and used < state['request']['max_queries']:
            return 'research'
        return 'finish'

    def research(state):
        used = sum(len(b.get('search_logs', [])) for b in state['batches'])
        open_pairs = [q for q in state['queries'] if any(f"{q['technology_id']}/{q['group']}" in gap for gap in state['gaps'])]
        queries = [{**q, 'id': q['id'] + ':retry', 'query': q['query'] + ' original source'} for q in (open_pairs or state['queries'])]
        return {'queries': queries[:state['request']['max_queries'] - used]}

    graph = StateGraph(StakeholderState)
    for name, node in {'plan': plan, 'search': search, 'extract': extract, 'review': review,
                       'research': research, 'revise': lambda s: {'revision_rounds': s['revision_rounds'] + 1},
                       'finish': lambda s: {'result': make_findings(s)}}.items():
        graph.add_node(name, node)
    graph.add_edge(START, 'plan')
    for a, b in [('plan', 'search'), ('search', 'extract'), ('extract', 'review'), ('research', 'search'), ('revise', 'extract'), ('finish', END)]:
        graph.add_edge(a, b)
    graph.add_conditional_edges('review', route, {x: x for x in ('research', 'revise', 'finish')})
    return graph.compile()


def run_stakeholder(request=None, technical_findings=None, backend=None):
    request = deepcopy(request or default_request())
    request.setdefault('max_queries', 8)
    if request.get('domain') not in ('datacenter', '데이터센터'):
        raise ValueError('평가 도메인은 데이터센터입니다.')
    date.fromisoformat(request['as_of_date'])
    if not 1 <= request['max_search_rounds'] <= 3 or not 0 <= request['max_revision_rounds'] <= 2 or not 1 <= request['max_queries'] <= 24:
        raise ValueError('검색 라운드 1~3, 수정 0~2, 전체 검색 질의 1~24 범위로 설정하세요.')
    if backend is None:
        from .backend import OpenAIBackend
        backend = OpenAIBackend()
    initial = {'request': request, 'technical_findings': deepcopy(technical_findings), 'queries': [], 'batches': [],
               'extraction': {'observations': [], 'gaps': []}, 'rounds': 0, 'revision_rounds': 0,
               'rejected': [], 'gaps': [], 'errors': [], 'result': {}}
    return build_stakeholder_graph(backend).invoke(initial, {'recursion_limit': 80})
