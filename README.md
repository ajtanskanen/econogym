# Finnish social security as an OpenAI Gym-environment

This library implements Finnish social security as an OpenAI Gym-environment that
can be used in a life-cycle model to predict, e.g., employment rate.

Description of the environment can be found (in Finnish!) from articles (Tanskanen, 2019a; Tanskanen, 2019b).

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


## Viittaukset

	@misc{econogym,
	  author = {Antti J. Tanskanen},
	  title = {Suomen sosiaaliturva ja verotus Gym-ympäristönä},
	  year = {2019a},
	  publisher = {GitHub},
	  journal = {GitHub repository},
	  howpublished = {\url{https://github.com/ajtanskanen/econogym}},
	}
	
Lyhyt kuvaus kirjastosta on myös julkaisussa
	
	@misc{lifecycle_rl_kak,
	  author = {Antti J. Tanskanen},
	  title = {Unelmoivatko robotit ansiosidonnaisesta sosiaaliturvasta?},
	  year = {2019b},
	  publisher = {},
	  journal = {KAK},
	  howpublished = {TBD},
	}		