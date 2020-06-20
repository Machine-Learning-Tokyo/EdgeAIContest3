from setuptools import setup

setup(
    name='edgeAIContest',
    version='1.0',
    description='A useful module',
    author='Man Foo',
    author_email='yashinde@gmail.com',
    packages=['edgeAIContest'],  # same as name
    # external packages as dependencies
    install_requires=['tensorflow==2.2.0'],
    scripts=[
        'scripts/cool',
        'scripts/skype',
    ]
)
