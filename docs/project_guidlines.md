## Page 1

&lt;img&gt;MBZUAI Logo&lt;/img&gt;
جامعة محمد بن زايد للذكاء الاصطناعي
MOHAMED BIN ZAYED UNIVERSITY OF ARTIFICIAL INTELLIGENCE

CV7502: Deep Learning for Visual Computing
Spring 2026

Project Guidelines Document

---

**Instructions:**

*   The project carries **significant weight (45%)** of the overall assessment grade, we highly recommend that students carefully read this project guidelines document.
*   Group Project. Maximum number of students per group: 4.
*   This project constitutes 45% of your total course marks. It has the following deliverables:
    *   Project proposal describing the problem setting and project plan. (5th week, 1 page) (20%)
    *   Interim progress report. (10th week, 4 pages) (20%)
    *   Project presentation (followed by Q/A session). (15th/16th week) (20%)
    *   Final project report and artefacts. (15th/16th week, 8 pages for the report) (40%)
*   Please note that for all reports, including proposal, mid-way and final, the main content should be in the mentioned page limit, and the references or supplementary information can go beyond the strict page limit to any number of pages.

---

---


## Page 2

# 1 Overview

A central goal of CV7502: Deep Learning for Visual Computing is to prepare students for advanced research and real-world problem solving in modern computer vision. This course project serves as a capstone experience that could integrate concepts from across the course, including convolutional networks, transformers, generative models, self-supervised learning, foundation models, and multimodal visual understanding. Through this project, students will have the chance to explore cutting-edge visual computing problems spanning images, videos, 3D data, or multimodal settings, and implement state-of-the-art deep learning techniques drawn from recent research literature. The project encourages students not only to apply existing methods, but also to analyze, adapt, and extend them in meaningful ways. While achieving improvements over existing benchmarks is welcome, it is not a strict requirement. Instead, emphasis is placed on depth of understanding, sound experimental design, and critical reasoning. Well-motivated ideas that do not yield performance gains are acceptable, provided students can clearly explain their hypotheses, experimental choices, and the reasons behind observed outcomes. Ultimately, students are expected to demonstrate a strong grasp of the problem domain, the relevant literature, modeling decisions, and evaluation methodology, and to communicate their insights clearly and professionally.

# 2 Project Instructions

*   The projects will be completed in groups of 3-4 students (but not more than 4 students). This is a group learning activity and individual projects will be discouraged. Please add group information in the Excel sheet.
*   Group members are responsible for equally dividing the work among them and making sure that each member contributes. This will be assessed during the final evaluation and via a peer-feedback survey at the end. We also expect a clear breakdown of tasks between team-members mentioned in the final-reports.
*   No unapproved extension of deadline is allowed. Late submission will lead to 0 credit.
*   LATEX typing is highly recommended. Typing with MS Word is also okay. Handwritten reports will not be given credit.
*   Explicitly mention your collaborators, if any. For the programming problem, it is absolutely not allowed to share your source code with anyone in the class as well as to use code from the Internet without reference.

All written articles must be in standard NeurIPS format. The page limits will be strictly followed. The following criterion will be used to evaluate research projects:

*   The novelty of the project ideas and applications. The groups are encouraged to come up with original ideas and novel applications for the projects. A project with new ideas (algorithms, methods, theory) in Deep Learning or new, interesting applications of existing algorithms is scored higher than a project without any new idea/application. If a project is built on an existing code-base, it must be clearly credited and differences should be explicitly stated.
*   The extensiveness of the study and experiments. A project that produces a more intelligent system by combining several Deep Learning techniques together, or a project that involves well-designed experiments and thorough analysis of the experimental results, or a project that nicely incorporates various real world applications, are scored higher.

---


## Page 3

- The writing style and clarity of the written reports. Clean and well-documented code will be scored higher.

## 3 Project Topic

Students are required to finalize a project topic prior to submitting the project proposal. Project topics should be aligned with the broad theme of deep learning for visual computing, and may involve one or more visual modalities such as images, video, 3D point clouds, or multimodal vision-language data. Projects typically fall into one or more of the following categories:

- **Application-oriented projects:** Apply deep learning models to a challenging real-world visual task (e.g., medical imaging, autonomous perception, video understanding, 3D scene analysis), with careful attention to data preprocessing, model design, and evaluation.
- **Methodological or algorithmic projects:** Propose a novel architectural modification, learning strategy, or combination of existing techniques (e.g., CNNs with transformers, self-supervised pretraining, foundation model adaptation) and study its impact on visual understanding tasks.
- **Analysis-driven:** Conduct in-depth analysis of existing models or learning paradigms, such as representation learning, robustness, generalization, or failure modes in modern vision systems.

Many strong projects naturally combine elements of application, methodology, and analysis. Students are strongly encouraged to draw inspiration from recent top-tier venues such as CVPR, ICCV, ECCV, NeurIPS, and ICML, and to ground their project in a clear understanding of prior work. A thorough literature review using tools such as Google Scholar is an essential first step. Dataset selection is a critical component of the project. Students should identify suitable public datasets early (e.g., vision benchmarks, medical datasets, video datasets, or 3D repositories) and avoid major dataset changes later unless strongly justified. Replicating results from a published paper can be a valuable learning exercise; however, projects should go beyond direct replication by incorporating extensions, ablations, new datasets, or deeper analysis. Simply running an existing codebase without critical engagement or modification is not sufficient.

## 4 Project Proposal (Weightage 20% & Due Date: Week 5)

A list of suggested projects and data sets are provided at the links above. Read the list carefully and decide a precise problem setting and dataset for your proposal. We would discourage changing the core datasets altogether later during the project. *Page limit:* Proposal main content should be one page maximum. Include the following information:

- Project title
- Data set to be used.
- Project idea. This should be approximately one-two paragraphs.
- Software you are planning to develop (mention core libraries you are planning to use).

---


## Page 4

*   Papers to read. Include 1-3 relevant papers. It will be best to carefully read at least one of them before submitting your proposal.
*   Teammates (if any) and work division. We expect projects done in a group to be more substantial than projects done individually.

Please refer to Figure [1] that illustrates the marks distribution for the project proposal.

<table>
<thead>
<tr>
<th colspan="7">Proposal [20]</th>
</tr>
</thead>
<tbody>
<tr>
<td>Intro<br>[5]</td>
<td>Motivation<br>[4]</td>
<td>Dataset<br>[2]</td>
<td>Literature Review<br>[4]</td>
<td>Work Division<br>[2]</td>
<td>Compute Load/Software<br>[3]</td>
<td>Total<br>[20]</td>
</tr>
</tbody>
</table>

Figure 1: Marks distribution for project proposal.

## 5 Midway Report (Weightage 20% & Due Date: Week 10)

This should be a 4 pages report in the form of a NeurIPS paper, and it serves as a check-point. By this date you should have applied one or more commonly used learning methods on a specific problem. The report should include a background section which describes the problem you are tackling, a related work section, a description of the method(s) you tried, an experiments section with detailed description of the dataset you experimented with and the results you obtained. For a theory project, instead of the experiments section, the report must include an insightful analysis of 2-3 related papers. The analysis should demonstrate your deep understanding of these papers. You can write this report in a way as you are writing some basic content in your final report. You can later reuse the content you put in Midway report to build your final report.

Please refer to Figure [2] that illustrates the marks distribution for the midway report.

<table>
<thead>
<tr>
<th colspan="7">Mid-Term Report [20]</th>
</tr>
</thead>
<tbody>
<tr>
<td>Background / Intro<br>[3]</td>
<td>Related Work<br>[3]</td>
<td>Proposed Method<br>[4]</td>
<td>Experiments / Analysis<br>[5]</td>
<td>Discussion / Future work<br>[3]</td>
<td>Quality of Writing / Formatting<br>[2]</td>
<td>Total<br>[20]</td>
</tr>
</tbody>
</table>

Figure 2: Marks distribution for midway report.

## 6 Final Report and Artefacts (Weightage 40% & Due Date: Week 15th/16th)

Around 15th/16th week, you will need to submit the final report via Moodle. A link to final report submission will be visible on Moodle close to that time. The report should have the following sections:

*   Background/Introduction (what is the problem you are solving? why is it important? why is it challenging?)
*   Related works (what has been done before in the relevant literature, what are close works?)

---


## Page 5

*   Method (what you have done in the project? provide a mathematical formulation of the ideas used in the project with appropriate description and visualizations if applicable)
*   Experiments and results (datasets and evaluation metrics used and the quantitative results obtained, presented in tables and/or plots, show qualitative results if applicable)
*   Artefacts must be uploaded on Github (version control software) and shared as a private repository. The repository must have:
    *   The code for the project and clear instructions on running the code,
    *   A readme file (text or markdown file) to specify the function of each component (file/folder) in the codebase,
    *   List all dependencies used in the project, and provide instructions on installing any dependencies (e.g., pytorch, cudatoolkit, scikit-learn, etc.)
    *   Provide a demo file available with sample inputs and outputs.
    *   Provide instructions on downloading data from publicly available links (for the datasets used in the project)
    *   If a project is built on an existing code-base, it must be clearly credited and differences should be explicitly stated in the readme file.

These details can be in a single readme file or in separate text files as convenient. Note that well-documented and clean code will be scored higher.

Please make sure you include a link to the Github private repo in the submitted report.

While you can include additional results/details in the appendix, make sure all the important details are in the main report submission and only the supplemental information is provided in the appendix.

Please refer to Figure 3 that illustrates the marks distribution for the final report and artefacts.

<table>
<thead>
<tr>
<th colspan="10">Final Report [40]</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Abstract</strong><br>[1]</td>
<td><strong>Intro</strong><br>[2]</td>
<td><strong>Contributions</strong><br>[1]</td>
<td><strong>Related Work</strong><br>[2]</td>
<td><strong>Prob. Statement</strong><br>[1]</td>
<td><strong>Proposed Method</strong><br>[5]</td>
<td><strong>Experimental Setup</strong><br>[4]</td>
<td><strong>Results &amp; Discussion</strong><br>[3]</td>
<td><strong>Limitations</strong><br>[0.5]</td>
<td><strong>Conclusion</strong><br>[0.5]</td>
<td><strong>Quality of Writing</strong><br>[2]</td>
<td><strong>Github Code and Readability</strong><br>[3]</td>
<td><strong>Total</strong><br>[25]</td>
<td><strong>Scaled</strong><br>[40]</td>
</tr>
</tbody>
</table>

Figure 3: Marks distribution for final report and artefacts. Zoom in for better viewing.

## 7 Project Presentation (Weightage 20% & Date: Week 15th/16th)

There will be a 10 minute project presentation in Week 15th/16th lecture slots (including 2 minutes for QnA). All project members should be present during the presentation. The session will be open to everybody.

The marking will be based on the clarity of the content, how well the ideas are communicated during the presentation, and answers to the questions asked during the presentation.

Please refer to Figure 4 that illustrates the marks distribution for project presentations.

---


## Page 6

<table>
  <thead>
    <tr>
      <th colspan="8">Presentation [20]</th>
    </tr>
    <tr>
      <th>Intro & Background</th>
      <th>Related Work</th>
      <th>Method</th>
      <th>Experiments & Results</th>
      <th>Time Management</th>
      <th>Aesthetics</th>
      <th>Total</th>
      <th>Scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>[2]</td>
      <td>[2]</td>
      <td>[4]</td>
      <td>[3]</td>
      <td>[2]</td>
      <td>[2]</td>
      <td>[15]</td>
      <td>[20]</td>
    </tr>
  </tbody>
</table>

Figure 4: Marks distribution for project presentations.

## 8 Frequently Asked Questions

1- Can two teams work on the same project? It is okay if two teams end up working on the same project as long as they do not coordinate to do so, to avoid any bias in the approach they take towards solving the problem. Alternatively, the teams can coordinate to make sure they work on different problems.
2- Are we required to use Python for the project? It is preferable to use Python since it will align with the course labs and TAs can help you with any questions. If you definitely prefer to use another language, it is allowed for the project.
3- Should the final project use only methods taught in the class? No, we don’t restrict you to only use methods/topics/problems taught in class. We encourage you to explore topics of your own interest and build on the concepts taught in the class to go above and beyond.