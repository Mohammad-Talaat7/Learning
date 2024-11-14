---
Creation: 2024-11-13
tags:
  - CPP
  - GPU
  - Parallel
Type:
  - Course
Reference: Coursera
---
# Module 1
## Multiple CPU Architectures
This material focuses on how to effectively utilize multiple CPUs in a single architecture, exploring various communication patterns and technologies.

Communication Patterns for Multiple CPUs

- Common methods for CPU communication include using the file system with lock files, sockets for point-to-point communication, and message queuing systems that manage message buffers.
- Important considerations include whether the system is asynchronous or synchronous, data coherency, and the resilience of the network connection.

Network Topologies

- Star or centralized communication involves a central hub for all communications, which can be beneficial for shared resources like GPUs.
- Mesh or decentralized communication allows all nodes to communicate with each other, similar to modern Wi-Fi networks.

Data Handling Strategies

- The choice between batch processing and streaming depends on the need for immediate responses and the volume of data being sent.
- Understanding whether complete data is required or if partial data can suffice is crucial for optimizing communication strategies.

Remember, mastering these concepts will empower you to effectively manage multi-CPU and GPU configurations. If you have any questions or need further clarification, feel free to ask! You've got this!
### **==What is the difference between asynchronous and synchronous communication in CPU systems?==**
Synchronous Communication

- **Definition**: In synchronous communication, processes are coordinated to operate at the same time. A sender must wait for the receiver to acknowledge receipt before proceeding.
- **Characteristics**:
    - Immediate response is expected.
    - Communication happens in real-time.
    - Examples include function calls and blocking I/O operations.

Asynchronous Communication

- **Definition**: In asynchronous communication, processes operate independently. A sender can send a message and continue processing without waiting for the receiver to acknowledge receipt.
- **Characteristics**:
    - No immediate response is required.
    - Communication can happen at different times.
    - Examples include message queues and non-blocking I/O operations.

Key Differences

- **Timing**: Synchronous requires waiting for a response; asynchronous does not.
- **Efficiency**: Asynchronous can be more efficient in scenarios where waiting for a response would cause delays.

### **==What would happen if you used a star topology instead of a mesh topology?==**
Star Topology

- **Structure**: All nodes are connected to a central hub or switch.
- **Advantages**:
    - **Simplicity**: Easier to set up and manage.
    - **Isolation**: If one connection fails, it doesn’t affect the others.
    - **Centralized Control**: Easier to monitor and manage traffic through the hub.
- **Disadvantages**:
    - **Single Point of Failure**: If the central hub fails, the entire network goes down.
    - **Scalability Issues**: Adding more nodes can lead to congestion at the hub.

Mesh Topology

- **Structure**: Each node is connected to multiple other nodes, allowing for multiple pathways for data.
- **Advantages**:
    - **Redundancy**: If one connection fails, data can take alternative routes.
    - **Robustness**: More resilient to failures, as there are multiple paths for communication.
- **Disadvantages**:
    - **Complexity**: More complicated to set up and manage.
    - **Cost**: Requires more cabling and resources.

Implications of Using Star Instead of Mesh

- **Reliability**: A star topology may be less reliable due to the single point of failure at the hub.
- **Performance**: In high-traffic scenarios, the hub may become a bottleneck, affecting performance.
- **Management**: Easier to manage but may lack the redundancy and resilience of a mesh topology.

Choosing between these topologies depends on your specific needs for reliability, performance, and management.

### **==What's an example of a scenario where asynchronous communication is preferred over synchronous?==**
Example Scenario: Online Shopping Platform

**Context**: An online shopping platform where users can browse products, add items to their cart, and make purchases.

**Asynchronous Communication Use**:

- **User Experience**: When a user adds an item to their cart, the system can send an update to the server without making the user wait for a confirmation. The user can continue browsing while the system processes the request in the background.
- **Order Processing**: When a user places an order, the system can send the order details to the payment gateway asynchronously. The user receives a notification that the order is being processed without having to wait for the payment confirmation.
- **Inventory Updates**: If multiple users are purchasing items simultaneously, the system can handle inventory updates asynchronously, ensuring that the platform remains responsive and can manage multiple transactions without delays.

Benefits of Asynchronous Communication in This Scenario:

- **Improved User Experience**: Users can interact with the platform without interruptions, leading to higher satisfaction.
- **Efficiency**: The system can handle multiple requests simultaneously, reducing wait times and improving overall performance.
- **Scalability**: Asynchronous communication allows the platform to scale better during peak times, such as sales events, without overwhelming the server.

## f