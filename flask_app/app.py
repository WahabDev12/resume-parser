# Import all packages
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import docx2txt
import os
from wtforms.validators import InputRequired
from os.path import join, dirname, realpath
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from pandas import json_normalize 
from collections import Counter
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import random
import string
import urllib
import base64

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import seaborn as sns

from pdfminer.high_level import extract_text
import nltk
import re
nltk.download('stopwords')



app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = join(dirname(realpath(__file__)), 'static/uploads/..')


@app.route("/test")
def api_test():
    return "Hello world" 


class UploadFileForm(FlaskForm):
    file = FileField("File", validators = [InputRequired()])
    submit = SubmitField("Submit")


PHONE_REG_GH = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')

PHONE_REG_USA = re.compile(r'/^\(?(\d{3})\)?[-]?(\d{3})[-]?(\d{4})$/') 

EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')


def extract_text_from_pdf(pdf_path):
    text = extract_text((pdf_path))
    return text

def extract_emails(resume_text):
    return re.findall(EMAIL_REG, resume_text)


def extract_names(txt):

    person_names = []

    for sent in nltk.sent_tokenize(txt):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                person_names.append(
                    ' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
                )

    return person_names 


# generate random file names
letters = [random.choice(string.ascii_lowercase) for i in range(4)]
random_filename = ''.join(letters)
    

def extract_phone_number(resume_text):
    phone = re.findall(PHONE_REG_GH, resume_text)

    if phone:
        number = ''.join(phone[0])

        if resume_text.find(number) >= 0 and len(number) < 16:
            return number
    return None 

file_skills_domain = pd.read_excel(join(dirname(realpath(__file__)), 'static/ResumeSkill.xlsx'))
file_skills_domain.columns = file_skills_domain.columns.str.strip().str.upper()

list_domains = []
for col in file_skills_domain.columns:
  
    file_skills_domain[col] = file_skills_domain[col].str.strip().str.upper()

    if col != 'EDUCATION' :
        list_domains.append('%s' % col)
        globals()['%s' % col]= [x for x in file_skills_domain[col].to_list() if type(x) != float]

list_skills = []
for i in list_domains:
  list_skills=list_skills + eval(i)


skills_dict = {}
domain_list = []


def extract_skills(input_text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)

    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]

    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]

    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 1, 3)))

    # we create a set to keep the results in.
    found_skills = set()

    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.upper() in list_skills:
            found_skills.add(token)

    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.upper() in list_skills:
            found_skills.add(ngram)

    for skill in found_skills :
      if skill.upper() not in skills_dict.keys():
          skill = skill.upper()
          cnt = 0
          for i in bigrams_trigrams:
              i = i.upper()
              if skill in i:
                  cnt += 1
                  for j in list_domains:
                    if skill in eval(j):
                      domain_list.append(j)

          print(skill.upper(), ' is repeated ' , cnt, ' times.')
          skills_dict[skill.upper()]= cnt




@app.route('/', methods = ['GET',"POST"])
def upload():
    root_dir = app.config['UPLOAD_FOLDER']
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data 
        file_path = ""
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),
        app.config['UPLOAD_FOLDER'], secure_filename(file.filename))) 
        
        file_path += ((os.path.join(root_dir,file.filename)))

        txt = extract_text_from_pdf(file_path)
        names = extract_names(txt)
        name_candidate = names[0] + ' ' + names[1].split(' ')[0]

        phone_number_gh = extract_phone_number(txt)
        phone_number_usa = None

        phone_contact = []
        phone_contact.append(phone_number_gh)

        print(phone_contact)
        
        emails = extract_emails(txt)

        if emails:
            print(emails)

        general_dict = { 'Name' : name_candidate.upper(),
              'Email' : emails ,
              'Contact' : phone_contact
    
        }  

        print(phone_contact) 
        extract_skills(txt) 

        new_vals = Counter(domain_list).most_common()
        new_vals = new_vals[::-1] #this sorts the list in ascending order
        print("this is: ", new_vals)

        domain_dict = {}
        for a, b in new_vals:
            domain_dict[a] = b


        general_dict["skills"] = skills_dict
        if len(new_vals) > 1:
            general_dict["domain"] = [new_vals[-1][0],new_vals[-2][0]]
        else: 
            new_vals = []

        skill_df = json_normalize(general_dict['skills'])
        num = skill_df.sum(axis = 1)[0]

        list_details = []
        for skill_x in skill_df.columns:
            list_details.append({
                    'doc':file_path ,'Name' : name_candidate.upper(),
                    'email' : emails[0] if emails else None,
                    'contact' : phone_contact[0] if phone_contact else None ,
                    'domain':[new_vals[-1][0][5:],new_vals[-2][0][5:]] if len(new_vals) > 1  else ["No domains"],
                    'skills':skill_x ,
                    'normalised_count':round((skill_df[skill_x][0]*100)/num,2),
                    'total_skills':list(skill_df.columns)
                })

        job_description = docx2txt.process(join(dirname(realpath(__file__)), 'static/job_desc.docx'))
        resume = docx2txt.process(join(dirname(realpath(__file__)), 'static/resume1.docx'))

        text = [resume, job_description]

        # Calculate similarity score 
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text)
        print("\Similarity Score: ")
        print(cosine_similarity(count_matrix))

        # Count match percentage
        match_percentage = cosine_similarity(count_matrix)[0][1] * 100
        match_percentage = round(match_percentage, 2)
        print()
        print("The match percentage of the resume is ", match_percentage,"%")

        # First figure
        img_one = BytesIO()
        plt.figure(figsize = (8,4))
        skills_dict
        my_df = pd.DataFrame(skills_dict.items())
        ax = sns.barplot(x = 0, y = 1, data = my_df)
        ax.set(xlabel = 'Skills', ylabel = '% Proficiency', title = 'SKILLS PROFICIENCY OF CANDIDATE ')

        plt.savefig(img_one, format = 'png')
        plt.close()
        img_one.seek(0)
        plot_url = urllib.parse.quote(base64.b64encode(img_one.getvalue()))

        # Second figure
        plt.figure(figsize=(8,4))
        img_two = BytesIO()
        my_df = pd.DataFrame(domain_dict.items())
        ax = sns.barplot(x = 0, y = 1, data = my_df)
        ax.set(xlabel = 'Domain', ylabel = '% Score', title = 'FIELD SPECIALITY OF CANDIDATE')

        plt.savefig(img_two, format = 'png')
        plt.close()
        img_two.seek(0)
        plot_url_two = urllib.parse.quote(base64.b64encode(img_two.getvalue()))

        return render_template(
                'frontend.html', plot_url = plot_url, plot_url_two = plot_url_two, 
                user_data = general_dict,
                match_percentage = match_percentage , form = form
            )
        

    return render_template('frontend.html', form = form)


if __name__ == '__main__':
    app.run(debug = True)