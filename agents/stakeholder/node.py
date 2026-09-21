"""이해관계자 에이전트 노드: 부모 State ↔ 이해관계자 내부 State 변환.

부모 그래프가 아는 것은 make_node 하나뿐이다. 내부 검색·검증 반복은 subgraph.py 에 있다.
"""
from copy import deepcopy
from functools import partial

from agents.stakeholder.subgraph import default_request, run_stakeholder


def team_update(final, existing_evidence=None):
    result = final['result']
    claims = {c['claim_id']: c for c in result['claims']}
    findings = [{**p, 'id': cid, 'claim_id': cid, 'statement': claims[cid]['statement'],
                 'evidence_ids': claims[cid]['evidence_ids'], 'conditions': claims[cid]['conditions'],
                 'uncertainty': claims[cid]['uncertainty']}
                for p in result['positions'] for cid in p['claim_ids']]
    direct = {(p['technology_id'], p['group']) for p in result['positions'] if p['target_scope'] == 'selected_technology'}
    return {'stakeholder_eval': {'perspective': 'stakeholder', 'findings': findings,
                'evidence_ids': list(result['evidence_store']), 'limitations': result['completion']['gaps'] + result['completion']['errors'],
                'confidence': len(direct) / 8, 'completion': result['completion'], 'search_outcomes': result['search_outcomes']},
            'evidence_store': result['evidence_store'], 'errors': result['completion']['errors']}


def stakeholder_agent(state, *, backend=None):
    request = default_request(state.get('as_of_date'))
    request['domain'] = state['domain']
    for side in ('sw', 'hw'):
        request[side] = deepcopy(state['selected_tech'][side])
    # 다른 관점 결과나 공유 evidence를 전달하지 않고 공통 기술 프로필만 투영한다.
    return team_update(run_stakeholder(request, state.get('tech_profiles'), backend))


def legacy_stakeholder_agent(state, *, backend=None):
    request = deepcopy(state['request'])
    request['domain'] = 'datacenter'
    return {'stakeholder_findings': run_stakeholder(request, state.get('technical_findings'), backend)['result']}


def make_node(backend=None, *, legacy=False):
    """부모 그래프에 등록할 노드 함수를 만든다.

    legacy=False: 팀 v0.3 EvaluationState(graph/team_state.py)용
    legacy=True : 이전 PipelineState(graph/state.py)용
    """
    return partial(legacy_stakeholder_agent if legacy else stakeholder_agent, backend=backend)
