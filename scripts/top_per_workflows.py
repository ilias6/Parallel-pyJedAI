from pyjedai.utils import (
                retrieve_top_workflows,
                to_path)

WORKFLOWS_PATH = to_path('~/pyJedAI/pyJedAI-Dev/script-results/sn-test.json')
STORE_PATH = to_path('~/pyJedAI/pyJedAI-Dev/script-results/best_workflows.json')


retrieve_top_workflows(workflows_path=WORKFLOWS_PATH,
                       store_path=STORE_PATH,
                       metric='recall',
                       top_budget=True)