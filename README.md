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

A version of the model with 15 states. This is the version used in article Tanskanen (2020)
Includes fixes to version v1.

### v3 - the last model with single individuals

A more realistic environment with 15 states. Includes fixes to versions v1 and v2.

### v4 - couples

A still more realistic environment with 16 states. It this version, couples are implemented.
Both partners in a couple make their decisions individually. This is the one that should be used.

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


The model is written in Python 3.

## References

	@misc{lifecycle_rl_,
	  author = {Antti J. Tanskanen},
	  title = {Elinkaarimalli},
	  year = {2019},
	  publisher = {GitHub},
	  journal = {GitHub repository},
	  howpublished = {\url{https://github.com/ajtanskanen/lifecycle_rl}},
	}

Description of the lifecycle model can be found from articles 
<a href='https://www.taloustieteellinenyhdistys.fi/wp-content/uploads/2020/06/KAK_2_2020_WEB-94-123.pdf'>Tanskanen (2020)</a> and 
<a href='https://www.sciencedirect.com/science/article/pii/S2590291122000171'>Tanskanen (2022)</a>.

    @article{tanskanen2022deep,
      title={Deep reinforced learning enables solving rich discrete-choice life cycle models to analyze social security reforms},
      author={Tanskanen, Antti J},
      journal={Social Sciences & Humanities Open},
      volume={5},
      pages={100263},
      year={2022}
    }
    
    @article{tanskanen2020tyollisyysvaikutuksien,
      title={Ty{\"o}llisyysvaikutuksien arviointia teko{\"a}lyll{\"a}: Unelmoivatko robotit ansiosidonnaisesta sosiaaliturvasta},
      author={Tanskanen, Antti J},
      journal={Kansantaloudellinen aikakauskirja},
      volume={2},
      pages={292--321},
      year={2020}
    }