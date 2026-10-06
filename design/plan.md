# Agent Plan

## Goal

A personalized job-search agent that discovers highly relevant jobs. It will first turn the markdown into the memory system, and then use the memory system to find the most relevant jobs for the user. The agent will also be able to access the internet to find additional information about job postings and companies. After finding the relevant jobs, the agent will provide url and description of the jobs to the user. At last, the agent will store the job information into the memory system for future reference and prevent the user from applying to the same job again.

## Agent Architecture

It will be based on Planner–Executor + Scout isolation.

## Main LLM model

deepseek-v4-flash: provide by deepseek api

## Memory System
I want to use the memory system in `/reference` directory, which is based on the HelloAgents framework. The memory system will be used to store and retrieve information about job search, including job postings, company information, and user preferences.

Qrant: will be used as vector storage, local host in docker

Neo4j: will be used as graph storage, local host in docker

SQLite: will be used as relational storage

text-embedding-v3: will be used for embedding generation, provide by aliyun api

## Agent Tools

Tavily: will be used as a tool for agent to access the internet, provide by tavily api

## Reference Documentation

My resume and job_preference in markdown format, store in `/my_information` directory

## Resources

`.env`: store the environment variables for the agent, including API keys and other sensitive information.

`docker-compose.yml`: define the services for Qrant and Neo4j.