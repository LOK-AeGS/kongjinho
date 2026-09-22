import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from langgraph.graph import StateGraph, START, END
from agents.stakeholder.node import stakeholder_agent, team_update
from agents.stakeholder.subgraph import GROUPS, default_request, run_stakeholder, validate_observations
from agents.stakeholder.backend import OpenAIBackend, unpack_search
from agents.stakeholder.models import Extraction, Observation
from agents.stakeholder.web import PageFetcher, digest, parse_html
from graph.team_state import EvaluationState, merge_evidence

URL='https://example.com/source'
TEXT='The deployment has operational benefits but migration requires careful planning.'

def observation(**updates):
    d=dict(technology_id='sw',target_name='DeepSeek-V2 MLA',target_scope='selected_technology',group='operator',speaker='Fixture engineer',affiliation=None,
           stance='positive',evidence_stance='support',domain_relevance='datacenter',statement='운영상 이점이 있다는 테스트 발언',source_url=URL,source_title='Fixture source',
           source_type='official_web',published_date='2026-01-01',primary_or_secondary='primary',direct_or_proxy='direct',page_or_locator='block:0001',quote=TEXT,
           conditions=[],uncertainty='합성 테스트 자료',bias_notes=[])
    d.update(updates)
    return Observation(**d)

def batch():
    blocks=[{'locator':'block:0001','text':TEXT}]
    page=dict(url=URL,final_url=URL,status='ok',blocks=blocks,metadata={'article:published_time':'2026-01-01'},content_hash=digest(json.dumps(blocks,ensure_ascii=False,sort_keys=True)),accessed_at='2026-09-21T00:00:00+00:00',snapshot_path=None)
    logs=[dict(id=f'{s}:{g}:r1',technology_id=s,group=g,query='fixture support counter neutral',status='ok',urls=[URL],provider='offline_fixture',as_of_date='2026-09-21') for s in ('sw','hw') for g in GROUPS]
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
        r=run_stakeholder(backend=FakeBackend([observation(),observation(group='supplier',statement='다른 해석')]))['result']
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
    def test_projection_and_parallel_merge(self):
        req=default_request(); state={'selected_tech':{s:req[s] for s in ('sw','hw')},'domain':'datacenter','tech_profiles':{'sw':{'method':'test'}},'market_eval':{'secret':'do not share'}}
        backend=FakeBackend(); update=stakeholder_agent(state,backend=backend)
        self.assertEqual(backend.calls[0][2],state['tech_profiles'])
        self.assertNotIn('secret',str(backend.calls))
        g=StateGraph(EvaluationState); g.add_node('a',lambda _:update); g.add_node('b',lambda _:{'evidence_store':update['evidence_store']}); g.add_node('join',lambda _:{})
        g.add_edge(START,'a'); g.add_edge(START,'b'); g.add_edge(['a','b'],'join'); g.add_edge('join',END)
        self.assertEqual(len(g.compile().invoke(state)['evidence_store']),1)
    def test_family_is_not_direct(self):
        f=run_stakeholder(backend=FakeBackend([observation(target_scope='technology_family')]))
        self.assertEqual(team_update(f)['stakeholder_eval']['confidence'],0)
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
                if url.endswith('/robots.txt'): return 200,{},b'User-agent: *\nAllow: /',url
                return 200,{'content-type':'text/html'},('<p>'+('original ' * 30)+'</p>').encode(),url
        with tempfile.TemporaryDirectory() as tmp:
            f=Fetcher(tmp,min_interval=0); r=f.fetch(URL)
            self.assertEqual(r['status'],'ok'); self.assertTrue(Path(r['snapshot_path']).exists()); self.assertEqual(f.fetch(URL),r)
        class Denied(Fetcher):
            def _get(self,url): return 200,{},b'User-agent: *\nDisallow: /',url
        with tempfile.TemporaryDirectory() as tmp: self.assertEqual(Denied(tmp).fetch(URL)['status'],'robots_denied')
    def test_extract_uses_original_not_notes(self):
        captured={}
        def parse(**kw):
            captured.update(kw); return SimpleNamespace(status='completed',output_parsed=Extraction(observations=[],gaps=[]))
        b=batch(); b['search_logs'][0]['discovery_notes']='DO_NOT_USE_SUMMARY'
        backend=OpenAIBackend(client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
        backend.extract(default_request(),[b],[])
        self.assertNotIn('DO_NOT_USE_SUMMARY',captured['input']); self.assertIn(TEXT,captured['input'])
    def test_search_requires_completed_search_action(self):
        r=SimpleNamespace(status='completed',output_text='',id='x',model_dump=lambda:{'output':[{'type':'web_search_call','status':'completed','action':{'type':'search','queries':['test'],'sources':[]}}]})
        self.assertEqual(unpack_search(r)['actions'][0]['queries'],['test'])
        r.status='incomplete'
        with self.assertRaises(ValueError): unpack_search(r)

if __name__=='__main__': unittest.main()
