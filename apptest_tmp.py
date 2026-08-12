from streamlit.testing.v1 import AppTest
import streamlit as st
print('streamlit:', st.__version__)
at = AppTest.from_file('app.py', default_timeout=300).run()
print('exception:', [e.value for e in at.exception] or 'none')
print('errors   :', [e.value for e in at.error] or 'none')
print('metrics  :', {m.label: m.value for m in at.metric})
print('tabs     :', len(at.tabs), '| dataframes:', len(at.dataframe))
for m in ['Decision Tree','Naive Bayes (Gaussian)','Random Forest (Ensemble)','SVM (RBF)','kNN']:
    a = AppTest.from_file('app.py', default_timeout=300).run()
    next(s for s in a.selectbox if m in s.options).select(m).run()
    print(f'  {m:26s} exc={[e.value for e in a.exception] or "none"}')
