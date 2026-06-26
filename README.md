# ABC Technologies Customer Automation System using LangGraph

## Project Overview

The **ABC Technologies Customer Automation System** is an AI-powered customer support automation project developed using **LangGraph**, **LangChain**, **OpenAI GPT**, and **SQLite**. The system intelligently classifies customer queries, routes them to the appropriate support agent, retrieves relevant knowledge using Retrieval-Augmented Generation (RAG), performs supervisor validation, supports Human-in-the-Loop (HITL) approval for critical requests, stores conversation history in SQLite, and recalls previous interactions when requested.

The project demonstrates how multiple AI agents can collaborate in a structured workflow to automate enterprise customer support while maintaining compliance and conversation memory.

---

## Features

* Intent Classification using AI
* Multi-Agent Routing

  * Sales Agent
  * Technical Support Agent
  * Billing Agent
  * Account Support Agent
* Retrieval-Augmented Generation (RAG)
* Human-in-the-Loop (HITL) Approval for Critical Billing Requests
* Supervisor Validation
* SQLite Conversation Memory
* Memory Recall
* Professional Final Response Generation
* Rich Console Output

---

## Project Workflow

```text
                    Customer Query
                           │
                           ▼
               Intent Classification
                           │
        ┌──────────┬──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼
     Sales     Technical   Billing    Account
      Agent       Agent      Agent      Agent
        │          │          │          │
        └──────────┴──────────┴──────────┘
                           │
                           ▼
                  RAG Knowledge Retrieval
                           │
          Critical Request? (Billing/Refund)
                    Yes             No
                     │               │
                     ▼               ▼
            Human-in-the-Loop      Continue
                     │
                     ▼
             Supervisor Validation
                     │
                     ▼
          Final Response Generation
                     │
                     ▼
         Store Conversation in SQLite
                     │
                     ▼
              Memory Recall (If Asked)
```

---

## Technologies Used

* Python 3.11
* LangGraph
* LangChain
* OpenAI GPT
* SQLite
* Rich
* FAISS / Vector Store (if used)
* Virtual Environment (venv)

---

## Project Structure

```text
ABC_Technologies_Project/
│
├── solution2.py
├── memory.db
├── requirements.txt
├── README.md
├── workflow_diagram.png
├── screenshots.pdf
└── knowledge_base/
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
```

### 2. Navigate to the Project Folder

```bash
cd ABC_Technologies_Project
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the following command:

```bash
python solution2.py
```

---

## Sample Execution

### Customer Query 1

```
Input:
What are the pricing plans available for your software?

Intent:
SALES

RAG:
Pricing Guide Retrieved

Output:
Starter Plan
Professional Plan
Enterprise Plan
```

---

### Customer Query 2

```
Input:
I forgot my account password.

Intent:
ACCOUNT

Output:
Password reset instructions generated.
```

---

### Customer Query 3

```
Input:
My application crashes whenever I upload a file.

Intent:
TECHNICAL

Output:
Technical troubleshooting steps generated.
```

---

### Customer Query 4

```
Input:
I need a refund for my annual subscription.

Intent:
BILLING

Output:
Human approval requested.
Supervisor validation completed.
Refund process initiated.
```

---

### Customer Query 5

```
Input:
What was my previous support issue?

Output:
Conversation history retrieved from SQLite memory.
```

---

## Key Components

### Intent Classification

Automatically classifies incoming customer requests into the appropriate support category.

Supported categories include:

* Sales
* Technical Support
* Billing
* Account Support
* Memory Recall

---

### RAG Retrieval

Relevant information is retrieved from the company's knowledge base before generating the response, ensuring accurate and context-aware answers.

---

### Human-in-the-Loop (HITL)

Critical billing and refund requests require manual approval before processing, ensuring compliance with company policies.

---

### Supervisor Validation

Every generated response passes through a supervisor validation stage before being returned to the customer.

---

### SQLite Memory

All customer interactions are stored in a SQLite database, enabling conversation history retrieval and contextual responses.

---

## Expected Outputs

The system demonstrates:

* Intent Classification
* Agent Routing
* RAG Context Retrieval
* Human-in-the-Loop Approval
* Supervisor Validation
* Memory Storage
* Memory Recall
* Final AI Response Generation

---

## Future Improvements

* Web-based user interface
* Voice assistant integration
* Multi-language support
* Real-time customer analytics dashboard
* Authentication and user management
* Cloud deployment using Docker and Kubernetes

---

## Author

**Name:** Your Name

**Project:** ABC Technologies Customer Automation System using LangGraph

**Technology Stack:** Python, LangGraph, LangChain, OpenAI GPT, SQLite
