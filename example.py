#!/usr/bin/env python3

"""
Example script demonstrating how to use the Vocal Analysis System.
This script analyzes a vocal performance and generates a detailed report.
"""

import os
from vocal_analyzer import VocalAnalyzer
from report_generator import ReportGenerator
from generate_report import generate_report_context

def main():
    # Path to your audio file
    audio_file = "path/to/your/vocal_performance.mp3"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        print("Please update the audio_file path to point to your vocal recording.")
        return

    try:
        # Initialize the vocal analyzer
        print("Initializing vocal analyzer...")
        analyzer = VocalAnalyzer()

        # Analyze the vocal performance
        print(f"Analyzing {audio_file}...")
        analysis = analyzer.analyze_file(audio_file)

        # Generate report context
        print("Generating report context...")
        context = generate_report_context(
            analysis=analysis,
            artist_name="Example Artist",
            performance_count=1
        )

        # Initialize report generator
        print("Initializing report generator...")
        rg = ReportGenerator()

        # Generate HTML report
        print("Generating HTML report...")
        html_report = rg.generate_html_report(
            context=context,
            output_html='reports/example_report.html'
        )
        print(f"HTML report generated: {html_report}")

        # Generate PDF report
        print("Generating PDF report...")
        pdf_report = rg.generate_pdf_report(
            html_path=html_report,
            output_pdf='reports/example_report.pdf'
        )
        print(f"PDF report generated: {pdf_report}")

        print("\nAnalysis complete! Check the reports/ directory for the results.")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 