#!/bin/bash
cd /workspaces/didactai
git add -A
git commit -m "Integrate neural student-state model into tutor flow

- Add automatic metric tracking during exercise sessions
- Implement neural model predictions directly in tutor page
- Add 'Progresul meu' page with statistics and progress tracking
- Create helper functions for neural integration
- Graceful fallback to rule-based feedback if model unavailable
- Preserve all existing sklearn models and exercise logic"
git push
