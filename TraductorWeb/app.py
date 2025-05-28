# app.py
"""
Main Flask application for the CSV File Translator.

This application provides a web interface for users to upload CSV files,
specify a column for translation (English to Spanish, logistics-focused),
and download the translated CSV. It utilizes a custom-trained AI model
for the translation, handled by the `ai_logic` module.

Features:
- File upload for CSVs.
- Specification of target column for translation by name or index.
- Specification of CSV delimiter.
- Background processing of translations to prevent HTTP timeouts for large files.
- Task status polling.
- Download of translated files.
- In-memory task and file management (suitable for demo/single-user scenarios).
"""

from flask import Flask, render_template, request, send_file, jsonify, url_for
import os
import csv
import io
import logging
from datetime import datetime
import uuid
import threading
import time

# --- AI Logic Integration ---
# Attempt to import core functions from the ai_logic module.
# These functions handle the loading of the translation model and the translation process itself.
try:
    from ai_logic import translateSingleTextDirect, loadGlobalTranslationModel
except ImportError as eImport:
    # If ai_logic cannot be imported, log a critical error and set functions to None.
    # The application will largely be non-functional without the AI component.
    logging.basicConfig(level=logging.CRITICAL) # Ensure basicConfig is called if logger isn't set up.
    logging.critical(f"CRITICAL App Init: Failed to import 'ai_logic'. Details: {eImport}. AI features will be disabled.")
    translateSingleTextDirect = None
    loadGlobalTranslationModel = None

# --- Flask Application Setup ---
app = Flask(__name__)

# Configure Flask's logger for application-level logging.
# In production (not debug mode), this sets up a handler for more structured logging.
if not app.debug:
    app.logger.setLevel(logging.INFO)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    app.logger.addHandler(streamHandler)

# Secret key for session management and other security-related features.
# os.urandom(24) generates a secure, random key.
app.config['SECRET_KEY'] = os.urandom(24)

# --- AI Model Initialization ---
# Load the AI translation model when the Flask application starts.
# This is done once to avoid reloading the model on every request, which would be slow.
if loadGlobalTranslationModel:
    try:
        app.logger.info("App Init: Attempting to load AI translation models...")
        loadGlobalTranslationModel() # Call the function from ai_logic.py
        app.logger.info("App Init: AI translation models loaded successfully.")
    except Exception as eLoadModel:
        app.logger.critical(f"CRITICAL App Init: Failed to load AI models during startup: {eLoadModel}", exc_info=True)
        # If model loading fails, core functionality is impaired.
        # Consider how to handle this (e.g., disable translation routes or show error page).
else:
    app.logger.error("CRITICAL App Init: 'loadGlobalTranslationModel' function is unavailable. AI logic module might have failed to import.")

# --- In-Memory Task and File Storage ---
# These dictionaries store the status of ongoing translation tasks and the content of translated files.
# Note: This is suitable for a demo or single-user application. For a production environment
# with multiple users or scalability needs, a more robust solution like a database or
# distributed cache (e.g., Redis) would be necessary for `tasksStatus` and a proper
# file storage solution (e.g., S3, local filesystem with a cleanup strategy) for `translatedFilesData`.
tasksStatus = {} # Stores {"taskId": {"status": "...", "progress": 0, "total": 0, "message": "..."}}
translatedFilesData = {} # Stores {"taskId": {"content": bytes, "filename": "..."}}

@app.context_processor
def injectNow():
    """Injects the current UTC datetime into template contexts.
    Allows templates to display dynamic information like a 'last updated' timestamp.
    """
    return {'now': datetime.utcnow()}

def processCsvTranslationInBackground(taskId: str, binaryFileContent: bytes, userColumnRef: str, csvDelimiter: str, originalFileName: str):
    """
    Processes a CSV file for translation in a background thread.

    Args:
        taskId (str): Unique identifier for this translation task.
        binaryFileContent (bytes): The raw byte content of the uploaded CSV file.
        userColumnRef (str): The user-specified column to translate (either name or 0-based index).
        csvDelimiter (str): The delimiter used in the CSV file (e.g., ',', ';').
        originalFileName (str): The original name of the uploaded file.
    
    Effects:
        - Updates `tasksStatus[taskId]` with progress and status.
        - Stores translated file content in `translatedFilesData[taskId]` upon completion.
        - Handles various errors including decoding, CSV parsing, and translation errors.
    """
    global tasksStatus, translatedFilesData # Allow modification of global task/file stores.
    try:
        app.logger.info(f"[Task {taskId}] Background processing started for: {originalFileName}")
        tasksStatus[taskId] = {"status": "processing", "progress": 0, "total": 0, "message": "Decoding file..."}

        # Attempt to decode the uploaded binary file content using common encodings.
        decodedContent = None
        usedEncoding = "unknown"
        # Common encodings for CSV files. 'utf-8-sig' handles UTF-8 with BOM.
        encodingsToTry = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        for enc in encodingsToTry:
            try:
                decodedContent = binaryFileContent.decode(enc)
                usedEncoding = enc
                app.logger.info(f"[Task {taskId}] File decoded successfully using '{usedEncoding}'.")
                break
            except UnicodeDecodeError:
                app.logger.debug(f"[Task {taskId}] Failed to decode with '{enc}'. Trying next.")
                continue
        
        if decodedContent is None:
            app.logger.error(f"[Task {taskId}] Failed to decode file with any of the attempted encodings.")
            raise ValueError("Could not decode the CSV file. Please ensure it's UTF-8 or Latin-1 encoded.")

        # Parse the decoded CSV content.
        # `io.StringIO` treats a string as a file. `newline=None` handles universal newlines.
        inputStream = io.StringIO(decodedContent, newline=None)
        allRows = list(csv.reader(inputStream, delimiter=csvDelimiter))
        inputStream.close()

        if not allRows:
            app.logger.warning(f"[Task {taskId}] CSV file is empty.")
            raise ValueError("The CSV file appears to be empty.")

        header = allRows[0]
        dataRows = allRows[1:]
        totalDataRows = len(dataRows)
        tasksStatus[taskId]["total"] = totalDataRows
        tasksStatus[taskId]["message"] = f"Preparing to translate {totalDataRows} data rows..."
        app.logger.info(f"[Task {taskId}] CSV Parsed. Header: {header}. Data rows: {totalDataRows}.")
        
        if totalDataRows == 0:
            app.logger.info(f"[Task {taskId}] No data rows to translate.")
            # Proceed to create an output file with just the header (and potentially new translation column)
            # This path is handled by the loop being empty.
        
        # Determine the index of the column to be translated.
        # The user can provide either a column name (string) or a 0-based index (integer).
        columnIndexToTranslate = -1
        try:
            # Attempt to convert userColumnRef to an integer (for index-based selection).
            columnIndexToTranslate = int(userColumnRef)
            if not (0 <= columnIndexToTranslate < len(header)):
                app.logger.error(f"[Task {taskId}] Column index {columnIndexToTranslate} is out of range for header length {len(header)}.")
                raise ValueError(f"Column index '{userColumnRef}' is out of range.")
        except ValueError:
            # If not an integer, assume it's a column name.
            try:
                columnIndexToTranslate = header.index(userColumnRef)
            except ValueError:
                app.logger.error(f"[Task {taskId}] Column name '{userColumnRef}' not found in header {header}.")
                raise ValueError(f"Column name '{userColumnRef}' not found in the CSV header.")
        
        app.logger.info(f"[Task {taskId}] Column to translate: '{header[columnIndexToTranslate]}' (Index: {columnIndexToTranslate}).")

        # Prepare for writing the output CSV.
        translatedRowsComplete = []
        outputHeader = list(header) # Create a mutable copy of the header.
        
        # Define the name for the new column that will store translations.
        fixedTranslationColumnName = "TRADUCCION" # "TRANSLATION" would be more English-consistent for internal code if desired

        # Check if a column named "TRADUCCION" already exists. If so, its content will be overwritten.
        # If not, append it as a new column.
        try:
            existingTranslationColumnIndex = outputHeader.index(fixedTranslationColumnName)
            app.logger.info(f"[Task {taskId}] Existing translation column '{fixedTranslationColumnName}' found at index {existingTranslationColumnIndex}. It will be overwritten.")
        except ValueError:
            outputHeader.append(fixedTranslationColumnName)
            existingTranslationColumnIndex = len(outputHeader) - 1 # Index of the newly added column.
            app.logger.info(f"[Task {taskId}] New translation column '{fixedTranslationColumnName}' will be added at index {existingTranslationColumnIndex}.")
        
        translatedRowsComplete.append(outputHeader) # Add the (potentially modified) header to output.

        # --- Main Translation Loop ---
        # Iterate through each data row, translate the specified cell, and update progress.
        if totalDataRows > 0:
            tasksStatus[taskId]["message"] = f"Translating {totalDataRows} rows..."
        translationLoopStartTime = time.time()

        for i, originalRow in enumerate(dataRows):
            dataRowNumber = i + 1 # For 1-based row numbering in logs/messages.
            rowProcessingStartTime = time.time()

            outputModifiedRow = list(originalRow) # Create a mutable copy of the current row.
            originalCellText = ""
            
            # Safely get the text from the specified column.
            # Handles cases where a data row might be shorter than the header.
            if 0 <= columnIndexToTranslate < len(originalRow):
                originalCellText = str(originalRow[columnIndexToTranslate])
            else:
                app.logger.warning(f"[Task {taskId}] Row {dataRowNumber} (length {len(originalRow)}) is shorter than expected "
                                   f"for translating column index {columnIndexToTranslate}. Original text will be empty.")
            
            # Perform the translation using the AI logic module.
            translatedCellText = translateSingleTextDirect(originalCellText) if translateSingleTextDirect else "[AI_SERVICE_UNAVAILABLE]"
            
            # Ensure the output row is long enough to place the translated text.
            # This handles rows that might be shorter than the header + new translation column.
            while len(outputModifiedRow) <= existingTranslationColumnIndex:
                outputModifiedRow.append('') # Pad with empty strings if necessary.
            
            outputModifiedRow[existingTranslationColumnIndex] = translatedCellText
            translatedRowsComplete.append(outputModifiedRow)
            
            # Update task progress.
            tasksStatus[taskId]["progress"] = dataRowNumber
            if dataRowNumber % 10 == 0 or dataRowNumber == totalDataRows : # Update message less frequently for performance
                 tasksStatus[taskId]["message"] = f"Translated row {dataRowNumber} of {totalDataRows}..."
            
            rowProcessingEndTime = time.time()
            app.logger.debug(f"[Task {taskId}] Row {dataRowNumber} translated in {rowProcessingEndTime - rowProcessingStartTime:.3f}s.")

        translationLoopEndTime = time.time()
        if totalDataRows > 0:
            app.logger.info(f"[Task {taskId}] Translation of {totalDataRows} data rows completed in {translationLoopEndTime - translationLoopStartTime:.2f}s.")
        else:
            app.logger.info(f"[Task {taskId}] No data rows were present to translate.")


        # Convert the list of translated rows back into a CSV formatted string, then bytes.
        outputTextBuffer = io.StringIO()
        csvWriter = csv.writer(outputTextBuffer, delimiter=csvDelimiter)
        csvWriter.writerows(translatedRowsComplete)
        
        # Store the translated content in memory, encoded as UTF-8 with BOM (for Excel compatibility).
        translatedFilesData[taskId] = {
            "content": outputTextBuffer.getvalue().encode('utf-8-sig'),
            "filename": f"translated_ingles_{originalFileName}" # Consider a more descriptive prefix/suffix.
        }
        outputTextBuffer.close()

        # Mark the task as completed.
        tasksStatus[taskId]["status"] = "completed"
        tasksStatus[taskId]["message"] = "Translation complete! Your file is ready for download."
        tasksStatus[taskId]["progress"] = totalDataRows # Ensure progress reflects total.
        app.logger.info(f"[Task {taskId}] Translation successful. File '{translatedFilesData[taskId]['filename']}' ready for download.")

    except ValueError as ve: # Catch specific, common errors first.
        app.logger.error(f"[Task {taskId}] Value error during background processing: {ve}", exc_info=True)
        tasksStatus[taskId] = {"status": "error", "message": f"Processing Error: {str(ve)}",
                               "progress": tasksStatus.get(taskId, {}).get("progress", 0),
                               "total": tasksStatus.get(taskId, {}).get("total", 0)}
    except Exception as e: # Catch-all for unexpected errors.
        app.logger.error(f"[Task {taskId}] Unhandled exception during background processing: {e}", exc_info=True)
        tasksStatus[taskId] = {"status": "error", "message": f"An unexpected server error occurred: {str(e)}",
                               "progress": tasksStatus.get(taskId, {}).get("progress", 0),
                               "total": tasksStatus.get(taskId, {}).get("total", 0)}


@app.route('/', methods=['GET'])
def indexPage():
    """Renders the main landing page of the application (`index.html`)."""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translateCsvRoute():
    """
    Handles CSV file uploads for translation.

    Expects a POST request with 'multipart/form-data' including:
    - 'archivo_csv': The CSV file to translate.
    - 'columna_a_traducir': The name or 0-based index of the column to translate.
    - 'delimitador_csv': The CSV delimiter (optional, defaults to ',').

    Responses:
    - 200 OK: With JSON {"taskId": "...", "statusUrl": "..."} if task is queued.
    - 400 Bad Request: If input is invalid (e.g., missing file, bad format, missing column).
    - 500 Internal Server Error: If the AI translation service is unavailable.
    """
    # Check if the AI translation service is operational.
    if not translateSingleTextDirect:
        app.logger.error("Translate Route: AI translation service (translateSingleTextDirect) is not available.")
        return jsonify({"error": "Translation service is currently unavailable. Please try again later."}), 500

    # Validate file presence and name.
    if 'archivo_csv' not in request.files:
        return jsonify({"error": "No CSV file selected. Please choose a file to upload."}), 400
    
    file = request.files['archivo_csv']
    if file.filename == '':
        return jsonify({"error": "No file selected or file has no name."}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Invalid file format. Only .csv files are accepted."}), 400

    # Validate required form fields for translation parameters.
    userColumnRef = request.form.get('columna_a_traducir', '').strip()
    csvDelimiter = request.form.get('delimitador_csv', ',').strip()
    if not csvDelimiter: csvDelimiter = ',' # Default to comma if empty.
    if not userColumnRef:
        return jsonify({"error": "Please specify the column to be translated (name or index)."}), 400

    # Generate a unique task ID for tracking.
    taskId = str(uuid.uuid4())
    originalFileName = file.filename
    # Read the entire file content into memory. For very large files, streaming would be more memory-efficient
    # but adds complexity to background processing and re-reading.
    binaryFileContent = file.stream.read()

    # Start the CSV processing in a new background thread.
    # This allows the server to respond quickly to the request while the (potentially long)
    # translation process runs independently.
    processingThread = threading.Thread(target=processCsvTranslationInBackground,
                              args=(taskId, binaryFileContent, userColumnRef, csvDelimiter, originalFileName))
    processingThread.start()
    
    # Initialize task status.
    tasksStatus[taskId] = {"status": "queued", "progress": 0, "total": 0, "message": "Translation task has been queued..."}
    app.logger.info(f"New translation task initiated. ID: {taskId}, File: {originalFileName}, Column: '{userColumnRef}', Delimiter: '{csvDelimiter}'")
    
    # Return task ID and URL to poll for status.
    return jsonify({"taskId": taskId, "statusUrl": url_for('getTaskStatus', taskId=taskId, _external=True)})

@app.route('/status/<taskId>', methods=['GET'])
def getTaskStatus(taskId: str):
    """
    Provides the status of a given translation task.

    Path Parameters:
        taskId (str): The unique ID of the translation task.

    Responses:
    - 200 OK: With JSON detailing task status, progress, message, and download URL if completed.
    - 404 Not Found: If the taskId is invalid or not found.
    """
    statusInfo = tasksStatus.get(taskId)
    if not statusInfo:
        app.logger.warning(f"Status request for unknown or invalid taskId: {taskId}")
        return jsonify({"error": "Task not found or ID is invalid."}), 404
    
    responseData = {
        "status": statusInfo["status"],
        "progress": statusInfo["progress"],
        "total": statusInfo.get("total", 0), # Use .get for 'total' as it might not be set initially.
        "message": statusInfo["message"]
    }
    
    # If the task is completed, include the download URL for the translated file.
    if statusInfo["status"] == "completed":
        responseData["downloadUrl"] = url_for('downloadTranslatedFile', taskId=taskId, _external=True)
        
    return jsonify(responseData)

@app.route('/download/<taskId>', methods=['GET'])
def downloadTranslatedFile(taskId: str):
    """
    Allows downloading of the translated CSV file.

    Path Parameters:
        taskId (str): The unique ID of the translation task.

    Responses:
    - 200 OK: With the translated CSV file as an attachment.
    - 404 Not Found: If the file for the taskId is not found, not ready, or task was erroneous.
    """
    fileDataInfo = translatedFilesData.get(taskId)
    taskInfo = tasksStatus.get(taskId)

    # Validate that the task exists, is completed, and file data is available.
    if not taskInfo or taskInfo.get("status") != "completed" or not fileDataInfo:
        app.logger.warning(f"Download request for incomplete or non-existent task/file. TaskId: {taskId}")
        return jsonify({"error": "File not found, not yet ready for download, or task encountered an error."}), 404

    # Prepare the file content for sending.
    # `io.BytesIO` creates an in-memory binary stream from the stored bytes.
    outputBytesBuffer = io.BytesIO(fileDataInfo["content"])
    outputBytesBuffer.seek(0) # Reset stream position to the beginning.
    
    app.logger.info(f"Download initiated for task {taskId}, filename: {fileDataInfo['filename']}")
    return send_file(
        outputBytesBuffer,
        mimetype='text/csv',
        as_attachment=True, # Prompts the browser to download the file.
        download_name=fileDataInfo["filename"] # Sets the default filename for the download.
    )

if __name__ == '__main__':
    # Entry point for running the Flask development server.
    # `debug=True` enables Werkzeug's debugger and reloader.
    # `use_reloader=False` is often recommended when using background threads in dev to avoid issues,
    # though modern Flask/Werkzeug might handle this better. Test for your specific setup.
    # `host='0.0.0.0'` makes the server accessible on your network, not just localhost.
    app.logger.info("Starting Flask development server...")
    # For production, use a proper WSGI server like Gunicorn or uWSGI.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)