# Finnish social security as an OpenAI Gym-environment

This library implements life-cycle of an individual s an OpenAI Gym-environmen.
can be used in a life-cycle model to predict, e.g., employment rate.

The library depends on the benefits library that implements Finnish social security a

Description of the environment can be found from articles (Tanskanen, 2019; Tanskanen, 2020).

## Versions

### v0 - minimal

Minimal environment with only three states: working, unemployed, retired.
This is the version used in article Deep reinforced learning enables solving discrete-choice life cycle models to analyze social security reforms (2020)

### v1 - baseline

A more realistic environment with only ten states. This is the version used in article 
Ty{\"o}llisyysvaikutuksien arviointia teko{\"a}lyll{\"a}: Unelmoivatko robotit ansiosidonnaisesta sosiaaliturvasta.

### v2 - improved

A more realistic environment with 15 states. This is the version used in article Tanskanen (2020b)
Includes fixes to version v1.

### v3 - the latest

A more realistic environment with 15 states. This is the version used that will be used in future articles.
Includes fixes to versions v1 and v3. This is the one that should be used.

## Rewards

Reward is log utility of net income minus a constant representing the free time lost.

# Installation

Clone the repository and run command 

	pip install -e .

in the parent directory of gym_unemployment, where you can find setup.py file.


# Adding a new environment

An example file that can be modified is saved as gym_unemployment/envs/test_environment.py

To add a new environment consists of a few steps: (1) copy test_environment.py, (2) registering
the new environment with Gym, (3) Updating the library. Steps (2) and (3) can be done by 
replicating the way test environment is included:

Test environment is included in econogym module in file gym_unemployment/envs/__init__.py 
with line

	from gym_unemployment.envs.test_environment import TestEnv
	
But it is not yet registered as a Gym environment. This is accomplished in file
gym_unemployment/__init__.py with lines

	register(
		id='test-v1',
		entry_point='gym_unemployment.envs:TestEnv',
	)

Finally, to include a new environment in the module, one needs to install it with pip. Run
command 

	pip install -e .

in the parent directory of gym_unemployment, where you can find setup.py file.


## References

	@misc{econogym,
	  author = {Antti J. Tanskanen},
	  title = {Suomen sosiaaliturva ja verotus Gym-ympäristönä},
	  year = {2019a},
	  publisher = {GitHub},
	  journal = {GitHub repository},
	  howpublished = {\url{https://github.com/ajtanskanen/econogym}},
	}
	
The library is described in articles

    @article{tanskanen2020deep,
      title={Deep reinforced learning enables solving discrete-choice life cycle models to analyze social security reforms},
      author={Tanskanen, Antti J},
      journal={arXiv preprint arXiv:2010.13471},
      year={2020}
    }
    
    @article{tanskanen2020tyollisyysvaikutuksien,
      title={Ty{\"o}llisyysvaikutuksien arviointia teko{\"a}lyll{\"a}: Unelmoivatko robotit ansiosidonnaisesta sosiaaliturvasta},
      author={Tanskanen, Antti J},
      journal={Kansantaloudellinen aikakauskirja},
      volume={2},
      pages={292--321},
      year={2020}
    }