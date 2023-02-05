# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, religion, or sexual identity
and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming,
diverse, inclusive, and healthy community.

## Our Standards

Examples of behavior that contributes to a positive environment for our
community include:

* Demonstrating empathy and kindness toward other people
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Accepting responsibility and apologizing to those affected by our mistakes,
  and learning from the experience
* Focusing on what is best not just for us as individuals, but for the
  overall community

Examples of unacceptable behavior include:

* The use of sexualized language or imagery, and sexual attention or
  advances of any kind
* Trolling, insulting or derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or email
  address, without their explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

## Enforcement Responsibilities

Community leaders are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in
response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

Community leaders have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct, and will communicate reasons for moderation
decisions when appropriate.

## Scope

This Code of Conduct applies within all community spaces, and also applies when
an individual is officially representing the community in public spaces.
Examples of representing our community include using an official e-mail address,
posting via an official social media account, or acting as an appointed
representative at an online or offline event.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the community leaders responsible for enforcement at either
clefter@student.ubc.ca, keishav@student.ubc.ca, or yigit61@student.ubc.ca.
All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the
reporter of any incident.

## Enforcement Guidelines

Community leaders will follow these Community Impact Guidelines in determining
the consequences for any action they deem in violation of this Code of Conduct:

### 1. Correction

**Community Impact**: Use of inappropriate language or other behavior deemed
unprofessional or unwelcome in the community.

**Consequence**: A private, written warning from community leaders, providing
clarity around the nature of the violation and an explanation of why the
behavior was inappropriate. A public apology may be requested.

### 2. Warning

**Community Impact**: A violation through a single incident or series
of actions.

**Consequence**: A warning with consequences for continued behavior. No
interaction with the people involved, including unsolicited interaction with
those enforcing the Code of Conduct, for a specified period of time. This
includes avoiding interactions in community spaces as well as external channels
like social media. Violating these terms may lead to a temporary or
permanent ban.

### 3. Temporary Ban

**Community Impact**: A serious violation of community standards, including
sustained inappropriate behavior.

**Consequence**: A temporary ban from any sort of interaction or public
communication with the community for a specified period of time. No public or
private interaction with the people involved, including unsolicited interaction
with those enforcing the Code of Conduct, is allowed during this period.
Violating these terms may lead to a permanent ban.

### 4. Permanent Ban

**Community Impact**: Demonstrating a pattern of violation of community
standards, including sustained inappropriate behavior,  harassment of an
individual, or aggression toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within
the community.

---

## Code Contribution Guidelines

### Python Code Guidelines

- Will be updated soon

### Pushing work to the remote repository

- Any new features need to be pushed to a branch separate to the main branch
- However, changes to files other than those that contain code (e.g. Jupyter Notebooks / Python scripts) can be pushed directly to main (e.g. READMEs, raw and processed data), unless otherwise specified

**Steps:**

1. `git branch name_of_feature`
2. `git switch name_of_feature`
3. (do your coding and make your commits to this branch)
    1. `git add name_of_file_changed` (or `git add .`)
    2. `git commit -m “meaningful text of changes that have been made”`
4. `git push`
5. `git switch main` (go back to main to bring it back in sync with tracking the remote main)
6. `git reset HEAD^` (moving HEAD back up to origin/main to track the remote main branch to make sure new features correctly come in when pulling)
7. `git switch feature` (go back to the feature branch to continue doing work, or create a new branch from here if you proceed to develop a new branch)

### Accepting and integrating work onto the remote repository

- Once some work has been done on individual branches and we are ready to merge to the main branch, each of us will submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request?tool=webui) for our own branches
- Together, as a group, after reviewing the [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request?tool=webui) (by logging comments on the request on GitHub), we will accept the request and merge to main

**Steps:**

> Note: this example assumes that there are 3 branches total, which need to be merged with the main branch

1. `git rebase feature_branch_1  feature_branch_2`
2. `git rebase feature_branch_2  feature_branch_3`
3. `git rebase feature_branch_3  main`
4. `git pull —rebase; git push`

- If you just want to work within your branch and push directly from that branch to the remote origin / main branch, you will need to set your feature branch to track the remote main branch

**Steps:**

1. Make sure you are already on your feature branch (`git switch name_of_feature`)
2. `git branch -u origin/main`

### Managing divergences

- In the case where other group members have made commits while you were working on your own feature, you will need to pull those changes first and integrate them with your own work before pushing your changes to your branch or to main

---

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.0, available at
https://www.contributor-covenant.org/version/2/0/code_of_conduct.html.

Community Impact Guidelines were inspired by [Mozilla's code of conduct
enforcement ladder](https://github.com/mozilla/diversity).

[homepage]: https://www.contributor-covenant.org

For answers to common questions about this code of conduct, see the FAQ at
https://www.contributor-covenant.org/faq. Translations are available at
https://www.contributor-covenant.org/translations.
