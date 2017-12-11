import hyperopt as hp

def get_space():
    learn_rates = np.linspace(0, 1.0, num=20, dtype=np.float32)
    space = {
        'learn_rate': hyperopt.hp.choice('learn_rate', hp.uniform('learn_rate', 0, 1.0)
    }


def optimize():


