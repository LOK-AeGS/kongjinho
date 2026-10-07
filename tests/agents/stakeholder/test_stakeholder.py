import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from langgraph.graph import StateGraph, START, END
from agents.stakeholder.node import build_request, make_node, to_app_update
from agents.stakeholder.subgraph import GROUPS, search_outcomes, shorten_quote, default_request, run_stakeholder, validate_observations
from agents.stakeholder.backend import OpenAIBackend, error_detail, select_blocks, unpack_search
from agents.stakeholder.models import Extraction, Observation
from agents.stakeholder.web import PageFetcher, digest, parse_html
from graph.state import AppState, create_initial_state, merge_evidence_store
from agents.stakeholder.evidence import merge_evidence


URL='https://example.com/source'
TEXT='The deployment has operational benefits but migration requires careful planning.'

def observation(**updates):
    d=dict(technology_id='sw',target_name='DeepSeek-V2 MLA',target_scope='selected_technology',group='adopter',speaker='Fixture engineer',affiliation=None,
           stance='positive',evidence_stance='support',domain_relevance='datacenter',statement='운영상 이점이 있다는 테스트 발언',source_url=URL,source_title='Fixture source',
           source_type='official_web',published_date='2026-01-01',primary_or_secondary='primary',direct_or_proxy='direct',evidence_level='unknown',page_or_locator='block:0001',quote=TEXT,
           conditions=[],uncertainty='합성 테스트 자료',bias_notes=[])
    d.update(updates)
    return Observation(**d)

def batch():
    blocks=[{'locator':'block:0001','text':TEXT}]
    page=dict(url=URL,final_url=URL,status='ok',blocks=blocks,metadata={'article:published_time':'2026-01-01'},content_hash=digest(json.dumps(blocks,ensure_ascii=False,sort_keys=True)),accessed_at='2026-09-21T00:00:00+00:00',snapshot_path=None)
    logs=[dict(id=f'{s}:{g}:{k}',technology_id=s,group=g,query='fixture support counter neutral',status='ok',urls=[URL],provider='offline_fixture',as_of_date='2026-09-21') for s in ('sw','hw') for g in GROUPS for k in ('r1','r1c')]
    return {'pages':{URL:page},'search_logs':logs,'errors':[]}

class FakeBackend:
    def __init__(self,items=None,data=None):
        self.items=[observation()] if items is None else items
        self.data=data or batch()
        self.calls=[]
    def search(self,request,queries,technical):
        self.calls.append(deepcopy((request,queries,technical)))
        r=deepcopy(self.data); ids={q['id'] for q in queries}
        r['search_logs']=[q for q in r['search_logs'] if q['id'] in ids]
        return r
    def extract(self,*args): return Extraction(observations=self.items,gaps=[])

class Tests(unittest.TestCase):
    def test_counter_not_found_is_allowed_with_logs(self):
        r=run_stakeholder(backend=FakeBackend())['result']
        self.assertEqual(r['completion']['status'],'complete')
        self.assertTrue(all(o['status']=='not_found' and o['query_ids'] for o in r['search_outcomes'] if o['stance']=='counter'))
        self.assertEqual(len(r['evidence_store']),1)
    def test_no_results_has_no_fabricated_evidence(self):
        b=batch(); b['pages']={}
        for q in b['search_logs']: q.update(status='no_results',urls=[])
        r=run_stakeholder(backend=FakeBackend([],b))['result']
        self.assertFalse(r['claims'])
        self.assertTrue(all(o['status']=='not_found' for o in r['search_outcomes']))
    def test_access_failure_is_blocked(self):
        b=batch(); b['pages'][URL]['status']='paywall'
        for q in b['search_logs']: q['status']='access_incomplete'
        r=run_stakeholder(backend=FakeBackend([],b))['result']
        self.assertEqual(r['completion']['status'],'partial')
        self.assertTrue(all(o['status']=='blocked' for o in r['search_outcomes']))
    def test_quote_locator_hash_url_gate(self):
        for u in ({'quote':'invented'},{'page_or_locator':'block:9999'},{'source_url':'https://fake.example/'}):
            ok,bad=validate_observations(Extraction(observations=[observation(**u)],gaps=[]),[batch()],'2026-09-21')
            self.assertFalse(ok); self.assertTrue(bad)
        b=batch(); b['pages'][URL]['content_hash']='tampered'
        self.assertFalse(validate_observations(Extraction(observations=[observation()],gaps=[]),[b],'2026-09-21')[0])
    def test_date_numeric_domain_gates(self):
        for u in ({'published_date':'2027-01-01'},{'statement':'성능 35.7% 향상'},{'statement':'5 GB/s'},{'domain_relevance':'other'}):
            self.assertFalse(validate_observations(Extraction(observations=[observation(**u)],gaps=[]),[batch()],'2026-09-21')[0])
    def test_unverified_date_is_partial(self):
        r=run_stakeholder(backend=FakeBackend([observation(published_date='2026-02-01')]))['result']
        self.assertEqual(r['completion']['status'],'partial')
        self.assertIsNone(next(iter(r['evidence_store'].values()))['published_at'])
    def test_shared_quote_has_multiple_claim_links(self):
        r=run_stakeholder(backend=FakeBackend([observation(),observation(group='investor',statement='다른 해석')]))['result']
        self.assertEqual(len(r['evidence_store']),1)
        self.assertEqual(len(next(iter(r['evidence_store'].values()))['claim_ids']),2)
    def test_reducer_idempotence_associativity_and_collision(self):
        a=run_stakeholder(backend=FakeBackend())['result']['evidence_store']; before=deepcopy(a)
        b=deepcopy(a); e=next(iter(b.values()))
        e.update(claim_id='market:x',claim_ids=['market:x'],perspective='market',perspectives=['market'],bindings=[{'claim_id':'market:x','perspective':'market','stance':'neutral'}])
        c=deepcopy(b); next(iter(c.values()))['accessed_at']='2026-09-22'
        self.assertEqual(merge_evidence(a,a),a)
        self.assertEqual(merge_evidence(a,b),merge_evidence(b,a))
        self.assertEqual(merge_evidence(merge_evidence(a,b),c),merge_evidence(a,merge_evidence(b,c)))
        self.assertEqual(a,before)
        next(iter(b.values()))['quote']='collision'
        with self.assertRaises(ValueError): merge_evidence(a,b)
    def test_query_budget_and_unsearched_status(self):
        req=default_request(); req['max_queries']=1; backend=FakeBackend([])
        r=run_stakeholder(req,backend=backend)['result']
        self.assertEqual(len(backend.calls),1)
        self.assertEqual(len(backend.calls[0][1]),1)
        self.assertTrue(any(o['status']=='unsearched' for o in r['search_outcomes']))
    def app_state(self, **extra):
        req=default_request()
        state=create_initial_state(
            request={'as_of':'2026-09-21','language':'ko','scope':'t','max_search_rounds':2},
            selected_tech={s:{'name':req[s]['name'],'short_name':s,'technology':s,'approach':'','source_ids':[],'selection_reason':'t'} for s in ('sw','hw')},
            corpus_manifest=[])
        state.update(extra)
        return state
    def test_node_returns_appstate_keys_and_projects_input(self):
        state=self.app_state(technical_findings={'claims':[{'text':'기술 주장'}],'secret_field':'do not share'},market_findings={'secret':'do not share'})
        backend=FakeBackend(); update=make_node(backend)(state)
        self.assertEqual(set(update),{'stakeholder_findings','evidence_store','search_log_by_perspective','quality_by_perspective'})
        self.assertNotIn('secret',str(backend.calls))
        self.assertEqual(backend.calls[0][2],{'claims':['기술 주장']})
        f=update['stakeholder_findings']
        self.assertEqual(f['perspective'],'stakeholder'); self.assertTrue(f['claims'] and f['records'])
        self.assertTrue(all(e['perspective']=='stakeholder' and e['quote'] for e in update['evidence_store'].values()))
        g=StateGraph(AppState); g.add_node('a',lambda _:update); g.add_node('b',lambda _:{'evidence_store':update['evidence_store']}); g.add_node('join',lambda _:{})
        g.add_edge(START,'a'); g.add_edge(START,'b'); g.add_edge(['a','b'],'join'); g.add_edge('join',END)
        self.assertEqual(len(g.compile().invoke(state)['evidence_store']),1)
    def test_non_datacenter_domain_is_rejected(self):
        with self.assertRaises(ValueError): make_node(FakeBackend())(self.app_state(domain='ondevice'))
    def test_rework_hint_targets_pairs_and_focus(self):
        state=self.app_state(rework_hint={'extra_rounds':1,'focus_queries':['공식 발표'],'gaps':[{'technology':'hw','criterion':'투자·산업 관계자'}]})
        state['request']['max_search_rounds']=3
        r=build_request(state)
        self.assertEqual(r['max_search_rounds'],3); self.assertEqual(r['only_pairs'],[['hw','investor']])
        backend=FakeBackend([]); make_node(backend)(state)
        self.assertEqual([q['id'] for q in backend.calls[0][1]],['hw:investor:r1','hw:investor:r1c'])
        self.assertIn('공식 발표',backend.calls[0][1][0]['query'])
    def test_hallucinated_opinions_never_reach_evidence_store(self):
        fake=[observation(quote='LLM이 지어낸 인용문'),observation(page_or_locator='block:9999',statement='다른 지어낸 발언'),
              observation(source_url='https://fake.example/x',statement='가짜 출처'),observation(published_date='2030-01-01',statement='미래 날짜')]
        update=make_node(FakeBackend(fake))(self.app_state())
        self.assertFalse(update['evidence_store']); self.assertFalse(update['stakeholder_findings']['claims'])
        self.assertEqual(update['stakeholder_findings']['status'],'partial')
        self.assertTrue(update['stakeholder_findings']['limitations'])
        self.assertEqual(update['quality_by_perspective']['stakeholder']['status'],'needs_review')
    def test_not_found_is_gap_without_fake_evidence(self):
        b=batch(); b['pages']={}
        for q in b['search_logs']: q.update(status='no_results',urls=[])
        f=make_node(FakeBackend([],b))(self.app_state())['stakeholder_findings']
        self.assertFalse(f['claims']); self.assertEqual(f['status'],'partial')
        self.assertTrue(all(g['reason'].startswith('not_found') for g in f['gaps']))
    def test_family_is_not_direct(self):
        f=make_node(FakeBackend([observation(target_scope='technology_family')]))(self.app_state())['stakeholder_findings']
        self.assertEqual(f['records'][0]['scope'],'class'); self.assertEqual(f['records'][0]['basis'],'inferred')
        self.assertIn('기술 계열',f['claims'][0]['limitations'][-1])
    def test_merge_with_parent_reducer_is_idempotent(self):
        store=make_node(FakeBackend())(self.app_state())['evidence_store']
        self.assertEqual(merge_evidence_store(store,store),store)
    def test_errors_are_redacted(self):
        class Bad(FakeBackend):
            def search(self,*args): raise RuntimeError('SECRET')
        f=run_stakeholder(backend=Bad()); self.assertNotIn('SECRET',str(f)); self.assertEqual(f['result']['completion']['status'],'failed')
    def test_parser_and_fetch_cache(self):
        blocks,meta=parse_html('<meta name="date" content="2026-01-01"><p>Hello</p><script>bad()</script><p>World</p>')
        self.assertEqual([b['text'] for b in blocks],['Hello','World'])
        self.assertEqual(blocks[1]['locator'],'block:0002')
        class Fetcher(PageFetcher):
            def _get(self,url):
                return 200,{'content-type':'text/html'},('<p>'+('original ' * 30)+'</p>').encode(),url
        with tempfile.TemporaryDirectory() as tmp:
            f=Fetcher(tmp,min_interval=0); r=f.fetch(URL)
            self.assertEqual(r['status'],'ok'); self.assertTrue(Path(r['snapshot_path']).exists()); self.assertEqual(f.fetch(URL),r)
    def test_extract_uses_original_not_notes(self):
        captured={}
        def parse(**kw):
            captured.update(kw); return SimpleNamespace(status='completed',output_parsed=Extraction(observations=[],gaps=[]))
        b=batch(); b['search_logs'][0]['discovery_notes']='DO_NOT_USE_SUMMARY'
        backend=OpenAIBackend(client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
        backend.extract(default_request(),[b],[])
        self.assertNotIn('DO_NOT_USE_SUMMARY',captured['input']); self.assertIn(TEXT,captured['input'])
    def test_extract_input_is_capped_and_reasoning_is_low_for_reasoning_models(self):
        captured={}
        def parse(**kw):
            captured.update(kw); return SimpleNamespace(status='completed',output_parsed=Extraction(observations=[],gaps=[]))
        big=batch(); blocks=[{'locator':f'block:{i:04d}','text':'x'*900} for i in range(40)]
        big['pages'][URL].update(blocks=blocks,content_hash=digest(json.dumps(blocks,ensure_ascii=False,sort_keys=True)))
        for model,expect in (('gpt-5-mini',True),('gpt-4.1-mini',False)):
            captured.clear()
            out=OpenAIBackend(model,client=SimpleNamespace(responses=SimpleNamespace(parse=parse))).extract(default_request(),[big],[])
            sent=json.loads(captured['input'])['pages'][URL]['blocks']
            self.assertLessEqual(sum(len(b['text']) for b in sent),6000)
            self.assertEqual(captured['max_output_tokens'],6000)
            self.assertEqual('reasoning' in captured,expect)
            self.assertTrue(any('입력 예산 초과' in g for g in out.gaps))
    def test_search_runs_queries_concurrently_and_keeps_order(self):
        import threading, time
        active=[0]; peak=[0]; lock=threading.Lock()
        def create(**kw):
            q=json.loads(kw['input'])['query']
            with lock:
                active[0]+=1; peak[0]=max(peak[0],active[0])
            time.sleep(0.3)
            with lock: active[0]-=1
            ann=[{'type':'url_citation','url':f"https://{q['id'].replace(':','-')}.example.com/p"}]
            return SimpleNamespace(status='completed',output_text='n',id='r-'+q['id'],model_dump=lambda ann=ann:{'output':[
                {'type':'web_search_call','status':'completed','action':{'type':'search','sources':[]}},
                {'type':'message','content':[{'annotations':ann}]}]})
        class Fetcher:
            def fetch(self,url): return {'status':'ok','url':url,'blocks':[],'metadata':{},'content_hash':'','final_url':url,'accessed_at':'x','snapshot_path':None}
        with tempfile.TemporaryDirectory() as tmp:
            backend=OpenAIBackend('gpt-5-mini',client=SimpleNamespace(responses=SimpleNamespace(create=create)),fetcher=Fetcher(),cache_dir=tmp); backend.pace_seconds=0
            queries=[{'id':f'sw:g{i}:r1','technology_id':'sw','group':'competitor','query':f'q{i}'} for i in range(6)]
            t=time.monotonic(); out=backend.search(default_request(),queries,None); took=time.monotonic()-t
        self.assertGreaterEqual(peak[0],2)
        self.assertEqual([l['id'] for l in out['search_logs']],[q['id'] for q in queries])
        self.assertEqual(len(out['pages']),6); self.assertFalse(out['errors'])
        self.assertLess(took,6*0.3+1.0)
    def test_long_quote_is_shortened_not_rejected(self):
        long_text=' '.join(f'w{i}' for i in range(60))
        blocks=[{'locator':'block:0001','text':long_text}]
        b=batch(); b['pages'][URL].update(blocks=blocks,content_hash=digest(json.dumps(blocks,ensure_ascii=False,sort_keys=True)))
        ok,bad=validate_observations(Extraction(observations=[observation(quote=long_text,statement='장문 발언')],gaps=[]),[b],'2026-09-21')
        self.assertEqual(len(ok),1); self.assertFalse(bad)
        self.assertLessEqual(len(ok[0].quote.split()),20); self.assertIn(ok[0].quote,long_text)
        self.assertIn('줄임',ok[0].uncertainty)
        self.assertEqual(shorten_quote('가 '*100).count('가'),20)
    def test_rejection_blocks_only_its_own_pair(self):
        b=batch()
        for q in b['search_logs']:
            q['urls']=[URL] if q['id'].startswith('sw:competitor') else []
            q['status']='ok'
        b['pages']={URL:b['pages'][URL]}
        r=run_stakeholder(backend=FakeBackend([observation(group='competitor',quote='지어낸 인용')],b))['result']
        by={(o['technology_id'],o['group'],o['stance']):o['status'] for o in r['search_outcomes']}
        self.assertEqual(by[('sw','competitor','support')],'blocked')   # 기각이 난 쌍
        self.assertEqual(by[('hw','investor','support')],'not_found')   # 영향 없는 쌍은 not_found
        self.assertEqual(by[('sw','adopter','counter')],'not_found')
    def test_partial_access_still_counts_as_completed_search(self):
        b=batch()
        for q in b['search_logs']: q['status']='access_incomplete'
        obs=search_outcomes([],[b])
        self.assertTrue(all(o['status']=='not_found' for o in obs))
        b['pages'][URL]['status']='paywall'
        self.assertTrue(all(o['status']=='blocked' for o in search_outcomes([],[b])))
    def test_queries_are_per_group_and_two_per_pair(self):
        backend=FakeBackend([]); run_stakeholder(backend=backend)
        q=backend.calls[0][1]
        self.assertEqual(len(q),12)
        self.assertEqual({x['id'].split(':')[2] for x in q},{'r1','r1c'})
        texts={x['group']:x['query'] for x in q if x['technology_id']=='sw' and x['id'].endswith('r1c')}
        self.assertIn('GitHub issue',texts['adopter']); self.assertIn('analyst',texts['investor']); self.assertIn('versus',texts['competitor'])
        self.assertTrue(all('data center' in x['query'] for x in q))
    def test_unpack_search_tolerates_missing_fields(self):
        r=SimpleNamespace(status='completed',output_text='',id='x',model_dump=lambda:{'output':[
            {'type':'web_search_call','status':'completed','action':{'type':'search','sources':None}},
            {'type':'message','content':None},
            {'type':'message','content':[{'annotations':None},{'annotations':[{'type':'url_citation','url':'https://a.example/'}]}]}]})
        self.assertEqual(unpack_search(r)['cited_urls'],['https://a.example/'])
        r.model_dump=lambda:{'output':None}
        with self.assertRaises(ValueError): unpack_search(r)
    def test_search_failure_keeps_masked_detail_and_retries_once(self):
        calls=[]
        def create(**kw):
            calls.append(1); raise TypeError('bad key sk-abc123XYZ')
        with tempfile.TemporaryDirectory() as tmp:
            backend=OpenAIBackend('gpt-4.1-mini',client=SimpleNamespace(responses=SimpleNamespace(create=create)),cache_dir=tmp); backend.pace_seconds=0
            out=backend.search(default_request(),[{'id':'sw:g:r1','technology_id':'sw','group':'competitor','query':'q'}],None)
        log=out['search_logs'][0]
        self.assertEqual(len(calls),2); self.assertEqual(log['status'],'search_failed')
        self.assertEqual(log['error_type'],'TypeError'); self.assertNotIn('abc123',log['error_detail']); self.assertIn('***',log['error_detail'])
        self.assertNotIn('abc123',str(out['errors']))
    def test_block_selection_prefers_relevant_blocks_and_keeps_order(self):
        blocks=[{'locator':f'block:{i:04d}','text':('filler '*100) if i!=7 else 'MLA deployment issue in production '+'x'*400} for i in range(1,12)]
        chosen=select_blocks(blocks,['mla','issue'],1200)
        self.assertIn('block:0007',[b['locator'] for b in chosen])
        self.assertEqual([b['locator'] for b in chosen],sorted(b['locator'] for b in chosen))
        self.assertLessEqual(sum(len(b['text']) for b in chosen),1200)
    def test_extract_tells_model_which_pair_each_page_was_found_for(self):
        captured={}
        def parse(**kw):
            captured.update(kw); return SimpleNamespace(status='completed',output_parsed=Extraction(observations=[],gaps=[]))
        b=batch()
        OpenAIBackend(client=SimpleNamespace(responses=SimpleNamespace(parse=parse))).extract(default_request(),[b],[])
        page=json.loads(captured['input'])['pages'][URL]
        self.assertIn('sw/competitor',page['found_for']); self.assertIn('hw/investor',page['found_for'])
        self.assertIn('found_for',captured['instructions'])
    def test_rework_merge_keeps_earlier_claims_and_gap_reasons(self):
        from agents.stakeholder.node import merge_findings
        first=make_node(FakeBackend([observation(),observation(group='competitor',statement='경쟁 발언')]))(self.app_state())['stakeholder_findings']
        self.assertEqual(len(first['claims']),2)
        second=make_node(FakeBackend([observation(group='investor',statement='투자 발언')]))(self.app_state())['stakeholder_findings']
        merged=merge_findings(first,second,{('sw','investor')})
        self.assertEqual(len(merged['claims']),3)
        self.assertEqual({(r['technology'],r['criterion']) for r in merged['records']},{('sw','도입 기업·개발자'),('sw','경쟁 기술 진영'),('sw','투자·산업 관계자')})
        same=merge_findings(first,first,{('sw','adopter')})
        self.assertEqual(len(same['claims']),2)
        r=[x for x in same['records'] if x['criterion']=='도입 기업·개발자'][0]
        self.assertEqual(sum(r['stance_counts'].values()),2)  # 같은 쌍은 건수 합산
        self.assertEqual(merge_findings(None,first,set()),first)
    def test_node_merges_when_reworking(self):
        state=self.app_state()
        first=make_node(FakeBackend([observation(group='competitor',statement='경쟁 발언')]))(state)
        state['stakeholder_findings']=first['stakeholder_findings']
        state['rework_hint']={'extra_rounds':1,'focus_queries':[],'gaps':[{'technology':'hw','criterion':'투자·산업 관계자'}]}
        second=make_node(FakeBackend([observation(technology_id='hw',group='investor',statement='HW 투자 발언')]))(state)
        self.assertEqual(len(second['stakeholder_findings']['claims']),2)  # 1차 주장이 남아 있다
        self.assertEqual(len(second['quality_by_perspective']['stakeholder']['checked_claim_ids']),2)
    def test_search_requires_completed_search_action(self):
        r=SimpleNamespace(status='completed',output_text='',id='x',model_dump=lambda:{'output':[{'type':'web_search_call','status':'completed','action':{'type':'search','queries':['test'],'sources':[]}}]})
        self.assertEqual(unpack_search(r)['actions'][0]['queries'],['test'])
        r.status='incomplete'
        with self.assertRaises(ValueError): unpack_search(r)

if __name__=='__main__': unittest.main()
