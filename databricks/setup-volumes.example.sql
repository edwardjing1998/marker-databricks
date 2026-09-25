-- ADMIN-REVIEWED EXAMPLE ONLY. Does not create an Azure account, storage credential,
-- external location, or network rules. Confirm HNS=true and an authorized external
-- location covering the three paths. Reuse existing volumes; never overlap them.
-- Replace the catalog/schema/principal with values from your workspace.
CREATE SCHEMA IF NOT EXISTS education_rag.document_processing;

CREATE EXTERNAL VOLUME IF NOT EXISTS education_rag.document_processing.marker_source
LOCATION 'abfss://education@chat8gpteducationaccount.dfs.core.windows.net/source/';

CREATE EXTERNAL VOLUME IF NOT EXISTS education_rag.document_processing.marker_generated
LOCATION 'abfss://education@chat8gpteducationaccount.dfs.core.windows.net/generated/';

CREATE EXTERNAL VOLUME IF NOT EXISTS education_rag.document_processing.marker_state
LOCATION 'abfss://education@chat8gpteducationaccount.dfs.core.windows.net/_marker_jobs/';


-- Apply separately to parent Run as and GPU launcher identities if different.
-- GRANT USE CATALOG ON CATALOG education_rag TO `YOUR-JOB-PRINCIPAL`;
-- GRANT USE SCHEMA ON SCHEMA education_rag.document_processing TO `YOUR-JOB-PRINCIPAL`;
-- GRANT READ VOLUME ON VOLUME education_rag.document_processing.marker_source TO `YOUR-JOB-PRINCIPAL`;
-- GRANT READ VOLUME, WRITE VOLUME ON VOLUME education_rag.document_processing.marker_generated TO `YOUR-JOB-PRINCIPAL`;
-- GRANT READ VOLUME, WRITE VOLUME ON VOLUME education_rag.document_processing.marker_state TO `YOUR-JOB-PRINCIPAL`;
