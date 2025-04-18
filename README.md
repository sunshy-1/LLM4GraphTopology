## Large Language Models as Topological Structure Enhancers for Text-Attributed Graphs (LLM4GraphTopology)

This is the Pytorch implementation for our *DASFAA'25* paper: [**LLM4GraphTopology: Large Language Models as Topological Structure Enhancers for Text-Attributed Graphs**](https://arxiv.org/abs/2311.14324). 

## Abstract
<div style="text-align: justify;">
The latest advancements in large language models (LLMs) have revolutionized the field of natural language processing (NLP). Inspired by the success of LLMs in NLP tasks, some recent work has begun investigating the potential of applying LLMs in graph learning tasks. However, most of the existing work focuses on utilizing LLMs as powerful node feature augmenters, leaving employing LLMs to enhance graph topological structures an understudied problem. In this work, we explore how to leverage the information retrieval and text generation capabilities of LLMs to refine/enhance the topological structure of text-attributed graphs (TAGs) under the node classification setting. First, we propose using LLMs to help remove unreliable edges and add reliable ones in the TAG. Specifically, we first let the LLM output the semantic similarity between node attributes through delicate prompt designs, and then perform edge deletion and edge addition based on the similarity. Second, we propose using pseudo-labels generated by the LLM to improve graph topology, that is, we introduce the pseudo-label propagation as a regularization to guide the graph neural network (GNN) in learning proper edge weights. Finally, we incorporate the two aforementioned LLM-based methods for graph topological refinement into the process of GNN training, and perform extensive experiments on four real-world datasets. The overall framework is as follows:
<div> 
<br>

![Framework](fig/framework.png)


## The code will be released before May ...
