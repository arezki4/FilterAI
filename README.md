# FilterAI
LLM content filtering module

## To contribute
1. Create a fork repo then you can clone your fork on your machine with:
```
git clone git@github.com:'yourfork-repo'/FilterAI.git
```
2. Then add another Git repository as a remote repository to keep your fork synchronized with the original repository with:
```
cd FilterAI
git remote add upstream git@github.com:arezki4/FilterAI.git
```
3. To retrieve changes from the remote repository, use:
```
git fetch upstream
```
4. then you can create a branch in your fork repo
```
git branch branche_name
git checkout branche_name
```
5. To update your repo from the remote repo
```
git fetch upstream/Master
git merge upstream/Master
```
6. After pushing your code you should create a Pull request on github and put the zone manager as a reviewer in order to review the code and validate your changes.
   
7. If you're the manager of the zone you've changed, you'll still need to ask a contributor for a code review, as no code can be pushed to master without a code review.
