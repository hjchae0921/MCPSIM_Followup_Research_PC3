"""Check native requests and scheduling without API calls or real credentials."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.kg_version_comparison import api, worker
from experiments.glm_four_arm_20261114 import run as controller
import launch


class QwenCloudTests(unittest.TestCase):
    def test_native_low_request_has_no_openrouter_routing(self):
        config = dict(model='qwen3.8-27b', transport='qwencloud', temperature=0,
                      reasoning={'effort': 'low'}, max_output_tokens=65536)
        request = api.request(config, [{'role': 'user', 'content': 'Reply OK.'}])
        self.assertEqual(request['model'], 'qwen3.8-27b')
        self.assertEqual(request['temperature'], 0)
        self.assertEqual(request['reasoning_effort'], 'low')
        self.assertEqual(request['max_tokens'], 65536)
        self.assertEqual(request['extra_body'], {'enable_thinking': True})
        self.assertNotIn('tools', request)
        self.assertEqual(api.request(config, [], limit=4096)['max_tokens'], 4096)

    def test_launcher_uses_the_explicit_official_key_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.key'
            path.write_text('fixture-only')
            env = {'QWENCLOUD_API_KEY_FILE': str(path)}
            launch.credentials(env, 'qwencloud')
            self.assertEqual(env['QWENCLOUD_API_KEY'], 'fixture-only')
            self.assertNotIn('OPENROUTER_API_KEY', env)

    def test_official_jobs_do_not_depend_on_openrouter_credit(self):
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            task = dict(model='Official fixture', graph='KG', case='01_heatsink',
                        directory=str(home / 'records/actual-start'), experiment_timestamp='actual-start')
            protocol = {'tasks': [task], 'model_configs': {task['model']: {'transport': 'qwencloud'}},
                        'frozen_sha256': {}}
            def execute(task, config, protocol):
                result = {**task, 'status': 'execution_success'}
                api.write(Path(task['directory']) / 'result.json', result)
                return result
            with patch.object(controller, 'HERE', home), patch.object(controller, 'report', return_value={}), \
                 patch.object(controller, 'credit', side_effect=AssertionError('Must not query OpenRouter')), \
                 patch.object(worker, 'run_workflow', side_effect=execute) as run, \
                 patch.object(worker, 'execute_stage', worker.execute_stage):
                controller.run(protocol)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(api.read(home / 'controller.json')['status'], 'finished')


if __name__ == '__main__':
    unittest.main()
