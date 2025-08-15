# I2Connect Traffic Safety System

<!-- Semantic Metadata for Better Search -->
**Keywords**: traffic safety, intersection monitoring, evidence theory, risk assessment, collaborative intelligence, driver monitoring, sensor fusion, concept 1, concept 2, safety concept, comparative analysis
**Project Partners**: University of Skövde, Scania, Smart Eye, Viscando
**Technical Terms**: Dempster-Shafer theory, Basic Probability Assignments, BPA, uncertainty reasoning, belief functions
**System Components**: Risk Assessment Module, Sensor-Edge-Cloud Module, HMI Module, Simulation Interface
**Safety Concepts**: Concept 1, Concept 2, Straight Trajectory Safety, Actor-Focused Risk

## Foreword

This document provides a structured and evolving knowledge base for the I2Connect research and development project. I2Connect is a forward-looking initiative aimed at enhancing traffic safety at intersections using intelligent sensing, probabilistic reasoning, and adaptive feedback systems. It addresses both technological and human factors in complex driving environments.

Designed to serve multiple stakeholders—researchers, engineers, evaluators, and funding agencies—this document consolidates architectural blueprints, methodological insights, system components, and implementation timelines into a progressive, accessible structure. Each section deepens in specificity, making it suitable for both introductory readers and domain specialists.

What follows is a detailed yet cohesive breakdown of the system architecture, the sensor technologies in use, the core risk assessment methodology grounded in Evidence Theory, and the user-centered simulation and evaluation strategy. The document is continuously updated to reflect project status and serves as a semantic backbone for future integration with ontology-driven systems and LLM-powered knowledge agents.

---

## 1. Project Overview

**I2Connect Project Overview | Traffic Safety Monitoring System | Intersection Safety Enhancement**

I2Connect is a comprehensive traffic safety monitoring system that enhances road safety through intelligent data analysis, real-time decision-making, and multi-sensor fusion. It targets improved intersection safety by integrating internal driver monitoring with external traffic data and presenting adaptive feedback through human-machine interfaces (HMI).

The project addresses the complexity and uncertainty inherent in urban driving environments through the application of modular architecture, adaptive sensor integration, and evidence-based risk computation.

**Safety Standards Alignment**: I2Connect's safety architecture aligns with internationally recognized automotive safety standards. Its adaptive driver-support functionalities are best positioned within [SAE J3016 Levels 2–3](https://www.sae.org/standards/content/j3016_202104/), where partial automation supports drivers without full autonomy. Furthermore, the system's cooperative and cloud-enhanced capabilities reflect the ambitions of [UNECE Regulation No. 157](https://unece.org/transport/vehicle-regulations/wp29-regulations/automated-lane-keeping-systems-alks), which governs Automated Lane Keeping Systems (ALKS) and highlights the importance of connected, harmonized decision-making across vehicles and infrastructure.

---

## 2. System Architecture

**I2Connect System Architecture | Modular Design | Real-Time Processing Components**

I2Connect's system architecture is designed around modular, scalable, and real-time interoperable components. These components are logically grouped by function and layered across sensing, local processing, cloud-level reasoning, and user interaction. This section introduces the key functional modules that collaboratively support the project's safety intelligence pipeline.

I2Connect's system is composed of modular, interoperable components, each designed to process and respond to high-volume data in real-time:

### 2.1 Risk Assessment Module

**Risk Assessment Module | Evidence Theory Implementation | Safety Confidence Computation**

Risk is defined as the quantified uncertainty regarding potential harm to road users at intersections. The central hypothesis is that real-time fusion of internal (driver gaze) and external (traffic actor) data leads to more accurate and timely risk prediction.

This module computes safety confidence values using [Evidence Theory](https://en.wikipedia.org/wiki/Evidence_theory), specifically the [Dempster-Shafer framework](https://en.wikipedia.org/wiki/Dempster–Shafer_theory), to evaluate uncertain and incomplete sensor information. It integrates internal gaze data and external actor behaviors into a unified situational risk score in real time. If the calculated belief in an unsafe condition (BetP_unsafe) exceeds a calibrated threshold, context-aware alerts are triggered through the HMI.

#### Evidence Theory in Risk Assessment

**Evidence Theory Methodology | Dempster-Shafer Theory | Basic Probability Assignments**

Evidence Theory, also known as Dempster-Shafer theory, belief functions, or uncertainty reasoning with Basic Probability Assignments (BPA), forms the mathematical foundation of I2Connect's risk assessment approach.

Each subsystem (e.g., gaze tracking, actor sensing) contributes Basic Probability Assignments (BPAs) to classify states as safe, unsafe, or unknown. Unlike classical probability, Evidence Theory allows for modeling incomplete knowledge by accommodating partial ignorance. This capability has been strengthened through theoretical extensions—such as [Yager's interpretation of the Dempster-Shafer framework](https://www.sciencedirect.com/science/article/abs/pii/0020025587900077), which offers alternative rules for evidence combination—and has been applied in advanced tracking systems as discussed in [Design and Analysis of Modern Tracking Systems](https://books.google.se/books/about/Design_and_Analysis_of_Modern_Tracking_S.html?id=lTIfAQAAIAAJ&redir_esc=y). 

BPAs are combined using Dempster's rule to resolve conflicts and derive a consistent belief distribution. Factors such as gaze confidence, trajectory conflict, and relative positioning are weighted to enhance the contextual validity of each risk estimate.

#### Risk Calculation Method

**Risk Calculation Process | BPA Fusion | Alert Threshold Logic**

The process begins by converting sensor readings into belief structures—triples representing the likelihood of the environment being safe, unsafe, or unknown. These Basic Probability Assignments (BPAs) are normalized to ensure valid total belief mass. Next, evidence from multiple sources (e.g., driver gaze, road actor motion) is mathematically fused using Dempster's rule to handle agreement and conflict among sources. 

The resulting fused BPA is used to derive a final decision value indicating the degree of unsafety (BetP_unsafe), calculated as the belief in the unsafe state plus a fraction of the unknown state. This final score determines whether the system should trigger an alert. The system prioritizes cases where the driver is attentive but exposed to external threats (e.g., crossing pedestrians, occluded paths). It supports real-time threshold tuning and temporal smoothing (e.g., memory windows) for robustness during pilot testing and deployment. The result is a dynamic risk profile that informs both the alert system and cloud-level cooperative reasoning.

### 2.2 Sensor-Edge-Cloud Module

**Sensor-Edge-Cloud Architecture | Multi-Layer Processing | Collaborative Intelligence**

This module encompasses a layered architecture that starts at the sensor level and extends through edge computing to a cloud-based collaborative intelligence platform.

#### Sensor Layer

**Sensor Layer | Smart Eye Integration | Viscando Emulation | Driver Monitoring**

At the foundation, [Smart Eye](https://smarteye.se) sensors monitor driver gaze behavior, capturing visual attention metrics critical for internal awareness modeling. Simultaneously, [Viscando](https://viscando.com) emulators simulate external actors—such as vehicles, cyclists, and pedestrians—across intersection scenarios. These sensor streams provide synchronized input on both driver state and surrounding environment, forming the dual basis of the risk estimation pipeline.

#### Edge Layer

**Edge Processing | Raspberry Pi Implementation | Real-Time Computation**

Lightweight processors embedded in the vehicle, such as Raspberry Pi units, execute local pre-processing and feature extraction. This includes real-time computation of safety-relevant signals like time-to-collision, gaze deviation, and actor proximity. Through initial data fusion and prioritization, edge units reduce latency and bandwidth demands, ensuring that only semantically enriched data is transmitted further.

#### Cloud Layer

**Cloud Intelligence | Cooperative Vehicle Networks | V2X Integration**

The cloud environment serves as a collective intelligence hub. It draws on cooperative vehicle intelligence approaches such as those described in the [5GAA White Paper on C-V2X Use Cases](https://5gaa.org/5gaa-releases-white-paper-on-c-v2x-use-cases-methodology-examples-and-service-level-requirements/), which highlight the importance of data exchange for real-time situational awareness and collision prevention. It aggregates risk assessments (ETA—Evidence Theory Analysis) from multiple vehicles and infrastructure units. These assessments include confidence-weighted predictions about possible collisions and are shared back with ego vehicles to provide context-aware situational awareness.

The cloud continuously computes a cooperative unsafety level and standardizes HMI alerts across vehicles based on real-time traffic patterns. Additionally, the cloud supports centralized training of risk models, long-term storage of scenario logs, and validation of concept performance. It enables synchronization between testbeds and live deployments, ensuring alignment between simulated evaluations and real-world behavior.

### 2.3 HMI Module

**Human-Machine Interface | Visual Alerts | Auditory Warnings | Driver Interaction**

The HMI Module outputs visual and auditory warnings to the driver through a configurable Human-Machine Interface, which aligns with emerging standards such as [Euro NCAP's Driver Monitoring Systems protocols](https://www.euroncap.com/en/vehicle-safety/the-ratings-explained/driver-monitoring-systems/). Alerts adapt based on computed risk thresholds, proximity of road actors, and gaze engagement. Visual cues are themed through dynamic display settings that can be customized based on alert level, color schemes, and contextual factors.

The system uses real-time evaluation of risk to determine when an alert should be issued. If the calculated belief in an unsafe situation surpasses the alert threshold, the HMI activates with the appropriate warning. The feedback is calibrated to avoid alarm fatigue while ensuring timely intervention. The system supports visual and audio modalities and logs all alerts for post-analysis and refinement.

### 2.4 Simulation Interface

**Simulation Environment | Urban Traffic Scenarios | Testing Platform**

A high-fidelity test environment that replicates urban traffic scenarios involving intersections, driver behavior, and vulnerable road users. It integrates all system modules in a controlled, repeatable setting to evaluate the effectiveness of sensor fusion, risk scoring, and HMI feedback. The simulation platform enables scenario generation, real-time monitoring of safety scores, and configurable risk thresholds, making it essential for iterative system testing and validation.

**Simulation Design Principles**:
* **Real-Time Fusion**: All risk estimates are computed per-frame during each simulation tick or real-time run, ensuring the system remains responsive to dynamic road conditions.
* **Explainability**: The system visualizes fused safety scores during evaluation, supporting transparent reasoning and pilot feedback. This traceability is essential for both development and regulatory validation.
* **Modularity**: Each subsystem—risk assessment, sensing, HMI, and simulation—operates independently and communicates through well-defined interfaces, enabling easy adaptation, replacement, or extension without compromising overall functionality.

---

## 3. Safety Concepts

**Safety Concepts Overview | Concept 1 | Concept 2 | Comparative Analysis**

This section introduces the safety concepts at the core of I2Connect's approach to intersection risk mitigation. These concepts are implemented as design strategies, validated through simulation, and progressively deployed in real-world conditions.

### Safety Concept Definition

**Safety Concept Framework | Design Abstraction | Modular Implementation**

A **safety concept** in the context of I2Connect is a formalized design abstraction that integrates sensing strategies, driver monitoring, risk assessment logic, and alert mechanisms to manage safety-critical decisions at intersections. Each safety concept encapsulates a combination of hardware configurations, data processing pipelines, and user feedback protocols tailored to specific driving conditions—such as trajectory patterns, road layouts, and actor complexity.

Safety concepts are defined to be modular and testable, making them adaptable to both simulated and real-world environments. Each safety concept serves as a structured *proof of concept*—a demonstrative implementation that embodies the key objectives and technological hypotheses of the I2Connect project. Through simulation and field trials, these concepts validate the system's capability to assess risk, generate context-aware alerts, and support human-machine collaboration in increasingly complex traffic environments.

**Concept 1** and **Concept 2** are specific instantiations of this overarching safety concept framework. Each is aligned with a particular operational scenario and maturity stage, reflecting I2Connect's staged deployment model.

### 3.1 Safety Concept 1: Straight Trajectory Safety (2025)

**Concept 1 | Safety Concept 1 | Straight Trajectory Safety | Onboard Sensors**

**Safety Concept 1** (also referred to as "Concept 1") represents the first implementation phase of I2Connect's safety framework, focusing on fundamental intersection safety scenarios.

**Key Characteristics of Concept 1**:
* **Sensor Configuration**: Uses only onboard sensors for data collection and processing
* **Operational Focus**: Focuses on the vehicle's forward motion across intersections with varying degrees of obstruction and complexity
* **Testing Approach**: Simulated scenarios involve changing intersection geometries and randomized actor configurations to test alert fidelity
* **Deployment Timeline**: Targeted for implementation in 2025
* **Technical Approach**: Implements basic Evidence Theory algorithms for risk assessment in straight-line driving scenarios

**Concept 1 Technical Implementation**: The first safety concept prioritizes simplicity and reliability, establishing the foundational algorithms and sensor integration patterns that will be extended in subsequent concept iterations.

### 3.2 Safety Concept 2: Actor-Focused Risk (2026+)

**Concept 2 | Safety Concept 2 | Actor-Focused Risk | Advanced Gaze Correlation**

**Safety Concept 2** (also referred to as "Concept 2") builds upon the foundation established by Concept 1, introducing advanced features for complex traffic scenarios.

**Key Characteristics of Concept 2**:
* **Enhanced Capabilities**: Builds upon Concept 1 by incorporating gaze-object correlation
* **Advanced Monitoring**: Includes external actor monitoring (e.g., vehicles, pedestrians) and time-to-collision factors
* **Complex Scenarios**: Designed for complex maneuvers such as left or right turns
* **Deployment Timeline**: Targeted for implementation in 2026 and beyond
* **Technical Evolution**: Represents a significant advancement over Concept 1 in terms of sensor fusion and risk assessment sophistication

**Concept 2 Technical Implementation**: The second safety concept introduces object-specific probabilistic analysis and enhanced cloud-based data integration, enabling more nuanced risk assessment in complex intersection scenarios.

### 3.3 Comparative Analysis: Concept 1 vs Concept 2

**Comparative Summary | Concept 1 vs Concept 2 | Feature Comparison | Implementation Differences**

The following comparative analysis provides insight into how the concepts evolve to support different road geometries, data sources, and driver engagement scenarios:

| Feature                    | Concept 1 (Safety Concept 1)           | Concept 2 (Safety Concept 2)              |
| -------------------------- | --------------------------------------- | ------------------------------------------ |
| **Primary Scenario**       | Driving straight through intersections  | Turning maneuvers (left/right)            |
| **Gaze Analysis**          | Binary (on/off road)                   | Object-specific probabilistic analysis    |
| **Environment Modeling**   | General traffic complexity              | Actor-centric contextual focus            |
| **Data Sources**           | Vehicle sensors only                    | Vehicle + cloud data integration          |
| **Deployment Timeline**    | 2025 (Pre-production)                  | 2026+ (Post-validation)                   |
| **Technical Complexity**   | Fundamental implementation              | Advanced feature integration              |
| **Risk Assessment**        | Basic Evidence Theory                   | Enhanced probabilistic modeling           |

**Key Differences Between Concepts**:
* **Concept 1** serves as the foundation, implementing core safety features for straightforward intersection traversal
* **Concept 2** extends these capabilities to handle complex scenarios requiring advanced sensor fusion and cloud-based intelligence
* Both concepts share the same underlying Evidence Theory framework but differ in implementation complexity and feature scope

---

## 4. Project Milestones

**Project Timeline | Development Milestones | Implementation Schedule**

| Date                | Deliverable                                     | Associated Concept              |
| ------------------- | ----------------------------------------------- | ------------------------------- |
| May 22 2025         | Simulation-integrated concept tests            | Concept 1 validation           |
| June 16 2025        | Real-world integration demo with Scania        | Concept 1 demonstration        |
| Sept–Oct 2025       | Data collection (Concept 1)                    | Concept 1 field testing        |
| Feb 2026            | Track tests (Concept 2)                        | Concept 2 initial validation   |
| May 2026            | Final project report to Vinnova                | Complete project documentation  |

---

## 5. Collaborators

**Project Consortium | Organizational Partners | Technical Contributors**

### Project Partners and Organizational Stakeholders

**Organizations involved**: University of Skövde, Scania, Smart Eye, Viscando  
**Institutional partners**: academic institutions, automotive companies, technology providers  
**Project consortium members**: research universities, vehicle manufacturers, sensor technology companies

The I2Connect project consortium consists of the following collaborative partners and organizational stakeholders:

#### Academic Partners

* **[University of Skövde](https://www.his.se/)** (HiS): Academic partner responsible for simulator development and system architecture design. This research institution provides the foundational simulation framework and coordinates the overall technical architecture.

#### Industrial Partners

* **[Scania](https://www.scania.com/)**: Industrial partner providing real-world validation and operational requirements. As a leading vehicle manufacturer, Scania contributes practical implementation expertise and real-world testing capabilities for both Concept 1 and Concept 2 validation.

#### Technology Partners

* **[Smart Eye](https://smarteye.se/)**: Technology partner specializing in in-vehicle gaze and behavior monitoring systems. Smart Eye provides the driver monitoring sensors and gaze tracking technology essential for internal driver state assessment in both safety concepts.

* **[Viscando](https://viscando.com/)**: Technology partner responsible for traffic actor sensing and emulation systems. Viscando contributes external environment monitoring capabilities and traffic simulation technologies supporting the development of both Concept 1 and Concept 2.

**Consortium Collaboration**: These collaborative organizations form the core project team, with each partner contributing specialized expertise in their respective domains. The consortium combines academic research capabilities, industrial implementation experience, and cutting-edge sensor technologies to achieve the project's traffic safety objectives across all concept implementations.

---

## 6. Applications and Use Cases

**Application Domains | Use Cases | Implementation Areas**

* **Smart Intersection Safety Systems**: Real-world deployment of both Concept 1 and Concept 2 for intersection monitoring
* **Dynamic Advanced Driver Assistance Systems (ADAS)**: Adaptive configuration based on concept-specific algorithms
* **AI-guided Driver Feedback and Monitoring Systems**: Implementation of Evidence Theory-based risk assessment
* **Academic Research Studies**: Probabilistic sensor fusion and uncertainty reasoning research
* **Real-time Traffic Safety Assessment**: Collision prevention using concept-specific approaches

---

## 7. Knowledge Utilization and Integration

**Knowledge Management | System Integration | Research Applications**

This document supports multiple usage contexts across technical, academic, and institutional domains. It serves not only as a structured reference for stakeholders but also as a foundation for automated reasoning and semantic enrichment.

**Primary Applications**:
* **Stakeholder Briefings**: Project partner onboarding and concept explanation
* **Queryable Interfaces**: LLM-based assistants and knowledge systems integration
* **Ontology Integration**: Workflow generation and semantic tool compatibility
* **Technical Documentation**: Collaborative development and concept implementation
* **Academic Reference**: Evidence theory and sensor fusion research foundation

**Concept-Specific Documentation**: This document serves as the authoritative source for understanding the differences between Concept 1 and Concept 2, their respective implementation timelines, and their technical specifications for research and development purposes.