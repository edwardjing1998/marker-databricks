-- ADMIN-REVIEWED EXAMPLE ONLY; not run by deployment.
-- Replace catalog/schema/volume names and principal application ID before executing.
-- This grants only data access, not job/workspace/Run as permissions.
GRANT USE CATALOG ON CATALOG education_rag
  TO `YOUR_SERVICE_PRINCIPAL_APPLICATION_ID`;
GRANT USE SCHEMA ON SCHEMA education_rag.document_processing
  TO `YOUR_SERVICE_PRINCIPAL_APPLICATION_ID`;
GRANT READ VOLUME ON VOLUME education_rag.document_processing.marker_source
  TO `YOUR_SERVICE_PRINCIPAL_APPLICATION_ID`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME education_rag.document_processing.marker_generated
  TO `YOUR_SERVICE_PRINCIPAL_APPLICATION_ID`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME education_rag.document_processing.marker_state
  TO `YOUR_SERVICE_PRINCIPAL_APPLICATION_ID`;
