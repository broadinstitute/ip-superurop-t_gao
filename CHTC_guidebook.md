# CHTC Guidebook

*https://docs.google.com/document/d/1arRuX7-QuKWpS1xej4o_pZevHEmNcbl7WsQGi13qI8Q/edit*

This is a general guide for running jobs with the Center for High Throughput Computing (CHTC). The setup steps will need to be completed for every new repository or project you create but not for every job.

> See also: HTCondor Users’ Manual, HTCondor User Tutorial

## 0. registering for CHTC

Before using CHTC, you will need to apply for an account here. You will be contacted within a few days for a consultation/orientation session where you can clarify the types of resources you need through CHTC and learn about some ways to get started, such as the Hello World example here.

## 1. setting up for CHTC

In order to run a job on CHTC, you’ll need to make sure CHTC has all the files you need, and you’ll also need to create a few additional files to configure the job you want to run.

Some of the setup steps below require connecting to CHTC. To connect to CHTC, run ssh <your-chtc-username>@submit1.chtc.wisc.edu on your local terminal. You will be prompted to enter your CHTC password.

### a. transferring code or data files

First, transfer any code or data files you’ll need to CHTC. Without code, you can’t run a job; without data, a job won’t have anything to run on! If transferring datasets, ensure that they are zipped to reduce the size so as not to overwhelm the CHTC server.

#### i. GitHub for code files

To transfer your code files, you can upload your code to a GitHub repository and then clone that repository (repo) after sshing into CHTC.

Instructions on creating a GitHub repo can be found here; to commit changes, using the GitHub CLI (rather than web browser) may be worth getting used to, especially if you think you might want to commit changes from CHTC.

After you have pushed changes to your GitHub repo, you can clone your repository after connecting to CHTC via git clone; to update your cloned repo after new changes have been pushed, run git pull.

#### ii. Docker for environment dependencies

If your code requires the installation of any libraries, then you’ll want to use Docker to ensure those same dependencies are satisfied on both your local machine (where you likely have tested your code to make sure it’s bug-free) and on CHTC. You can do this by creating a Docker container on DockerHub.

[This article](https://phoenixnap.com/kb/how-to-commit-changes-to-docker-image) has a good example of the process of committing changes to a Docker image. (Note that there is a small typo on Step 2 of the article: instead of sudo docker run -it cf0f3ca922e0 bin/bash, the example should read sudo docker run -it cf0f3ca922e0 /bin/bash.)
sudo docker images: displays available containers
sudo docker ps -a: displays launched containers

Afterward, run docker push your-hub-username/repo-name[:tag], where [:tag] denotes an optional tag (defaults to latest); note that the name of the new image as you have committed it must be your-hub-username/repo-name in order for docker push to work!

You will also need to include information about your docker image in your .sub file.

#### iii. scp for data files

Your code may require data files such as image datasets. These files may be too large to upload via GitHub; instead, they can be transferred using SCP.

The command to transfer a file from path/to/data (on your local computer) to destination/path/to/data on CHTC is: scp path/to/data <yourchtcusername>@submit1.chtc.wisc.edu:destination/path/to/data. (There is no line break here, only a space and text wrap by Google Docs.)

If you are transferring contents of a directory, you’ll need to use the recursive version of the command (specified using the tag -r): scp -r local/path/to/data <yourchtcusername>@submit1.chtc.wisc.edu:destination/path/to/data

If you are transferring images from the Broad Institute server, the local path to your data may be difficult to find. On Mac, you can open the server in Finder and drag from the folder to your Terminal window to get the full path; for smb://hydrogen/imaging_analysis, the path will look something like /Volumes/imaging_analysis/...

### b. adding CHTC configuration files

Beyond your code, CHTC also needs .sub and .sh files so that it knows what you want to run and how you want it to be run. These files specify parameters of your job(s), including important information about input/output files.

#### .sub

This is the CHTC Condor submit file, which tells CHTC which files to transfer and how to run them.

You can specify additional files to transfer in order to run the job beyond the default. See the documentation for specifying which input files to transfer. Keep in mind that files will be transferred “flat” (to the same directory) unless preserve_relative_paths is set to True, so if your .py script references a file that will be transferred, then make sure the relative directory is correct.

If you need to pull a Docker image to run your script, then include the following lines in your .sub file:
```
universe = docker
docker_image = your-hub-username/repo-name:vX
```

Here, vX specifies which version number of the Docker image you want to pull. You can also omit it to default to the latest version.

After your job is run, you may want output files to be transferred back. See the documentation for specifying how to transfer output files.

#### .sh

This is the bash script called by your .sub file. When you type commands into a terminal, those are bash commands. The contents of the .sh script are what you would want to type into the terminal (e.g., python3 main.py -flag --argument) to run your code. You can also include echo commands to print from the console to the output files produced after the job has finished running.

## 2. running jobs on CHTC
Setting up to run jobs is probably the most difficult part of using CHTC; future tweaks to your .py file or other script probably won’t require large changes to your setup.

Running jobs requires only a call to condor_submit: after running ssh <your-chtc-username>@submit1.chtc.wisc.edu to connect to CHTC, cd to where your .sub file is located and submit your file to CHTC by running condor_submit <your-sub-file>.sub.

### HTCondor commands
Besides condor_submit, there are several other HTCondor commands you can use:

- condor_q: displays jobs you have submitted and their statuses. Jobs which are already completed may not appear in the queue.
- condor_q -hold your-job-id -analyze shows a description of a job that has been put on hold.
- condor_release your-chtc-username: releases all jobs you own that may be on hold. Occasionally, jobs will become placed on hold if there are errors. Try condor_release and if that doesn’t work, run condor_q -analyze your-job-id to try to diagnose why.
- condor_rm your-job-id: removes your job from the queue

> See also: Managing a Job — HTCondor Manual 9.3.0 documentation
