# Contribution Guidelines

**Configuration**  
1. Fork the repository so you can make changes without affecting the original product until you're ready to merge them.
2. Set the original repository as a remote.
`git remote add upstream https://github.com/wri/plantation_classifier.git`
`git fetch`
3. Verify the upstream repository you've specified for your fork.
`git remote -v`

**Making changes locally**  
1. Install or update to the packages specified in the [requirements.txt](https://github.com/wri/plantation_classifier/blob/master/requirements.txt) file.
2. Create a working branch for your changes.
`git checkout -b branch-name`

**Submit a pull request**
When you've completed your changes, create a pull request (PR). Don't forget to:
* Commit the changes to your branch. 
`git push origin branch-name`
* Create a pull request.
`git request-pull branch-name master`
* Provide detail in the template to help reviewers understand your changes and the purpose of the PR.
* Link the PR to an issue if you are solving one.
* If any changes are requested before the PR can be merged, make changes in your fork and then commit them to your branch.
* Use this [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge conflicts and other issues.