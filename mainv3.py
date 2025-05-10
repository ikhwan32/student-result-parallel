import time
import random
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Set random seed for reproducibility
# make sure data is identical across runs
random.seed(42)
np.random.seed(42)

# Ikhwan Hadi 2024793017
# Function to generate student data
def generate_student_data(num_students, num_courses_per_student):
    student_data = []
    
    for student_id in range(1, num_students + 1):
        student_name = f"Student_{student_id}"
        
        # Generate courses for each student
        for course_id in range(1, num_courses_per_student + 1):
            course_name = f"Course_{course_id}"
            
            # Generate assignment scores (assume 5 assignments per course)
            assignment_scores = [random.uniform(50, 100) for _ in range(5)]
            
            # Generate exam score
            exam_score = random.uniform(50, 100)
            
            # Add student data
            student_data.append({
                'student_id': student_id,
                'student_name': student_name,
                'course_id': course_id,
                'course_name': course_name,
                'assignment_scores': assignment_scores,
                'exam_score': exam_score
            })
    
    return pd.DataFrame(student_data)

# Function to calculate grade for a single student-course record
def calculate_grade(record):
    # Calculate average assignment score (60% of final grade)
    avg_assignment_score = np.mean(record['assignment_scores'])
    assignment_component = 0.6 * avg_assignment_score
    
    # Get exam score (40% of final grade)
    exam_component = 0.4 * record['exam_score']
    
    # Calculate final score
    final_score = assignment_component + exam_component
    
    # Determine letter grade
    if final_score >= 70:
        letter_grade = 'A'
        grade_point = 4.0
    elif final_score >= 60:
        letter_grade = 'B'
        grade_point = 3.0
    elif final_score >= 50:
        letter_grade = 'C'
        grade_point = 2.0
    elif final_score >= 40:
        letter_grade = 'D'
        grade_point = 1.0
    elif final_score >= 30:
        letter_grade = 'E'
        grade_point = 0.5
    else:
        letter_grade = 'F'
        grade_point = 0.0
    
    # Simulate some computational load (e.g., complex calculations)
    result = 0
    for i in range(1000000):  # Adjust this number to simulate computational load
        result += i % 10
    
    return {
        'student_id': record['student_id'],
        'student_name': record['student_name'],
        'course_id': record['course_id'],
        'course_name': record['course_name'],
        'final_score': final_score,
        'letter_grade': letter_grade,
        'grade_point': grade_point
    }

# Function to calculate GPA for a student
def calculate_student_gpa(student_grades):
    total_grade_points = sum(grade['grade_point'] for grade in student_grades)
    gpa = total_grade_points / len(student_grades) if student_grades else 0
    return gpa

# Function to process all students sequentially
def process_students_sequential(df):
    results = []
    
    # Calculate grades for each student-course record
    for _, record in df.iterrows():
        record_dict = record.to_dict()
        grade_info = calculate_grade(record_dict)
        results.append(grade_info)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate GPA for each student
    student_gpas = {}
    for student_id in results_df['student_id'].unique():
        student_grades = results_df[results_df['student_id'] == student_id].to_dict('records')
        student_gpas[student_id] = calculate_student_gpa(student_grades)
    
    return results_df, student_gpas

# Function to process all students in parallel
def process_students_parallel(df):
    # Convert DataFrame to list of dictionaries for parallel processing
    records = df.to_dict('records')
    
    # Use ProcessPoolExecutor for parallel processing
    num_processes = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(calculate_grade, records))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate GPA for each student
    student_gpas = {}
    for student_id in results_df['student_id'].unique():
        student_grades = results_df[results_df['student_id'] == student_id].to_dict('records')
        student_gpas[student_id] = calculate_student_gpa(student_grades)
    
    return results_df, student_gpas

# Ikhwan Hadi 2024793017
# Main function to run the simulation
def run_simulation(num_students, num_courses_per_student):
    print(f"Generating data for {num_students} students with {num_courses_per_student} courses each...")
    student_data = generate_student_data(num_students, num_courses_per_student)
    
    print(f"Total records: {len(student_data)}")
    
    # Process sequentially
    print("\nProcessing student grades sequentially...")
    start_time = time.time()
    seq_results, seq_gpas = process_students_sequential(student_data)
    seq_time = time.time() - start_time
    print(f"Sequential processing time: {seq_time:.2f} seconds")
    
    # Process in parallel
    print("\nProcessing student grades in parallel...")
    start_time = time.time()
    parallel_results, parallel_gpas = process_students_parallel(student_data)
    parallel_time = time.time() - start_time
    print(f"Parallel processing time: {parallel_time:.2f} seconds")
    
    # Calculate speedup factor
    speedup = seq_time / parallel_time
    print(f"\nSpeedup factor: {speedup:.2f}x")
    print(f"Parallel processing was {speedup:.2f} times faster than sequential processing")
    
    # Validate that both methods produced the same results
    print("\nValidating results...")
    seq_results_sorted = seq_results.sort_values(['student_id', 'course_id']).reset_index(drop=True)
    parallel_results_sorted = parallel_results.sort_values(['student_id', 'course_id']).reset_index(drop=True)
    
    results_match = seq_results_sorted.equals(parallel_results_sorted)
    print(f"Results match: {results_match}")
    
    # Create visualization
    plot_results(seq_time, parallel_time)
    
    return seq_time, parallel_time, speedup

# Function to plot the results
def plot_results(seq_time, parallel_time):
    plt.figure(figsize=(10, 6))
    
    # Bar chart for processing times
    plt.bar(['Sequential', 'Parallel'], [seq_time, parallel_time], color=['blue', 'green'])
    
    plt.title('Student Grade Processing: Sequential vs Parallel', fontsize=16)
    plt.ylabel('Processing Time (seconds)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add processing time labels on top of bars
    for i, time_val in enumerate([seq_time, parallel_time]):
        plt.text(i, time_val + 0.1, f'{time_val:.2f}s', 
                 ha='center', va='bottom', fontsize=12)
    
    plt.savefig('processing_comparison.png')
    plt.close()
    print("Results visualization saved as 'processing_comparison.png'")

# Run simulation with different dataset sizes
if __name__ == "__main__":
    # Medium dataset
    print("=" * 50)
    print("RUNNING SIMULATION WITH MEDIUM DATASET")
    print("=" * 50)
    run_simulation(num_students=100, num_courses_per_student=5)
    
    # Larger dataset (uncomment to run)
    # WARNING, TAKE ALMOST A MINUTE TO RUN THE SEQUENTIAL 
    #print("\n" + "=" * 50)
    #print("RUNNING SIMULATION WITH LARGER DATASET")
    #print("=" * 50)
    #run_simulation(num_students=200, num_courses_per_student=8)