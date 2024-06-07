import PyPDF2
import os
import docx
import pandas as pd
import openai
import json
from openai import OpenAI
import psycopg2
import shutil
from langchain_postgres.vectorstores import PGVector
from langchain_openai.embeddings import OpenAIEmbeddings
import uuid
from sqlalchemy import create_engine
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn


@app.post("/upload_resume/")
async def upload_files_and_extract_text(files: List[UploadFile] = File(...)):
    result_df = pd.DataFrame()
    for file in files:
        try:
          print(f"Filename: {file}")
          file_location = os.path.join(".", file.filename)

          with open(file_location, "wb") as buffer:
              shutil.copyfileobj(file.file, buffer)

          resume_text = read_documents(file_location)
          prompt_template = create_resume_extraction_prompt(resume_text)
          response = generate_response(prompt_template)

          print(f"Total tokens used: {response.usage.total_tokens}")

          gpt_data_data = response.choices[0].message.content
          formatted_data = convert_to_json(gpt_data_data)
          # print(f"Formatted data: {formatted_data}")
          candidate_df = pd.DataFrame([formatted_data])

          vector = candidate_df.summary.apply(lambda x: get_embedding(x)).values[0]

          cursor, engine, connection = get_connection()

          uu_id = "can-" + str(uuid.uuid4())

          candidate_df['candidate_id'] = uu_id
          candidate_df['valid_parse'] = True
          candidate_df.columns = ['name','country','place', 'email','mobileNumber','college','designation','totalExperience','companiesWorked','skillset','degree','domians','hourlyrate','noticeperiod','summary','candidate_id','valid_parse']
          candidate_df['resume'] = resume_text
          candidate_df.to_sql(name = CANDIDATE_TABLE, con = engine, if_exists='append', index = False)

          # print(f" dataframe: {candidate_df}")
          create_vector_query = f"CREATE TABLE IF NOT EXISTS {VECTOR_TABLE} (id VARCHAR PRIMARY KEY, embedding vector(1000));"
          insert_vector_query = f"INSERT INTO {VECTOR_TABLE} (id, embedding) VALUES (%s, %s);"
          vector_resume = candidate_df.resume.apply(lambda x: get_embedding(x)).values[0]
          exec_query(cursor, create_vector_query)
          exec_query(cursor, insert_vector_query, (uu_id, vector_resume,))
          connection.commit()
          cursor.close()
          connection.close()

          # candidate_df['']
          result_df = pd.concat([result_df, candidate_df], ignore_index=True, axis=0)

        except Exception as e:
          return {
              "status": "error",
              "code": 505,
              "error": {
                  "message": str(e),
                  "message1": formatted_data
              }
          }
    # list_of_dicts = result_df.to_dict(orient='records')
    # list_of_json = [json.dumps(record) for record in list_of_dicts]

    return {
              "status": "success",
              "code": 200,
              "data": result_df.to_dict(orient='records')
          }
@app.post("/upload_jd/")
async def upload_jd_def(job_description: str):
   try:
      # Query PostgreSQL to get matching profiles
      cursor, engine, connection = get_connection()
      prompt_template = create_jd_summary_prompt(job_description)
      response = generate_response(prompt_template)

      summary = convert_to_json(response.choices[0].message.content)['summary']

      uu_id = "jd-" + str(uuid.uuid4())
      jd_data = {
          "jd_id":uu_id,
          "summary": summary
      }
      jd_df = pd.DataFrame(jd_data, index=[0])
      jd_df.to_sql(name = JD_TABLE, con = engine, if_exists='append', index = False)
      return {
          "status": "success",
          "code": 200,
          "data": jd_data
      }
   except Exception as e:
      return {
          "status": "error",
          "code": 505,
          "error": {
              "message": str(e)
          }
      }


@app.post("/match_profiles/")
async def match_profiles_def(jd_id: str):
   try:
      # Query PostgreSQL to get matching profiles
      cursor, engine, connection = get_connection()
      query_get_embeddings = f"SELECT summary FROM {JD_TABLE} WHERE jd_id = '{jd_id}'"
      cursor.execute(query_get_embeddings)
      summary = cursor.fetchall()[0][0]
      # print(summary)
      embedding = get_embedding(summary)
      # print(embedding)
      query_similar_vectors = f"SELECT id, 1 - (embedding <=> '{embedding}') AS similarity_score FROM {VECTOR_TABLE} ORDER BY embedding <=> '{embedding}';"

      df_scores = pd.read_sql_query(query_similar_vectors, connection)
      # cursor.execute(query_similar_vectors)
      # similar_vectors = cursor.fetchall()

      # profiles = tuple([_[0] for _ in similar_vectors])
      # scores = tuple([_[1] for _ in similar_vectors])

      get_profiles_query = f"SELECT candidate_id, name, email, designation, summary from {CANDIDATE_TABLE} WHERE candidate_id IN {tuple(df_scores['id'])}"

      df_profiles = pd.read_sql_query(get_profiles_query, connection)

      df = pd.merge(df_profiles, df_scores, left_on = "candidate_id", right_on = "id", how = "inner")
      df = df.sort_values(by='similarity_score', ascending=False)
      df = df.head()
      df['jd_id'] = jd_id
      df['match_summary'] = """### Summary and Rating:
- Skills: 8.5/10
  - The resume matches most required skills, with partial matches on specific RedHat Linux and explicit OpenShift/Kubernetes troubleshooting details.
- Experience: 9/10
  - The experience duration and relevance are well-matched to the JD.
- Location: 10/10
  - The candidate is located in Bangalore, which is a perfect match.
  ### Overall Match Rating: 9/10
Rehan Khan's resume matches the job description very well, especially in terms of location and relevant experience. While there is a partial match in certain specific skills (like explicit RedHat Linux administration and detailed troubleshooting of OpenShift/Kubernetes), the overall fit is strong."""

      df = json.loads(df.to_json(orient='records'))

      return {
          "status": "success",
          "code": 200,
          "data": df
      }
   except Exception as e:
      return {
          "status": "error",
          "code": 505,
          "error": {
              "message": str(e)
          }
      }


@app.post("/generate_jd/")
async def generate_jd(job_description: str):
   try:
      prompt_template = create_jd_creation_prompt(job_description)
      response = generate_response(prompt_template)
      print(f"Total tokens used: {response.usage.total_tokens}")
      return {
            "status": "success",
            "code": 200,
            "data": response.choices[0].message.content
        }
   except Exception as e:
      return {
          "status": "error",
          "code": 505,
          "error": {
              "message": str(e)
          }
      }


uvicorn.run(app, port=8000)
