CREATE DATABASE CHANGE_MGMT_DASHBOARD;

--Tables
CREATE TABLE USERS (
	user_id INT AUTO_INCREMENT PRIMARY KEY,
	user_full_name VARCHAR(200) NOT NULL,
	user_email VARCHAR(200),
	user_role VARCHAR(100),
	create_date DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE CHANGE_TYPE (
	type_id INT AUTO_INCREMENT PRIMARY KEY,
	type_name VARCHAR(100) NOT NULL,
	type_description VARCHAR(500)
);

CREATE TABLE CHANGE_STATUS (
	status_id INT AUTO_INCREMENT PRIMARY KEY,
	status_name VARCHAR(100) NOT NULL,
	status_description VARCHAR(500)
);

CREATE TABLE PRIORITIES (
	priority_id INT AUTO_INCREMENT PRIMARY KEY,
	priority_name VARCHAR(100) NOT NULL,
	priority_level INT
);

CREATE TABLE change_request (
	change_id BIGINT AUTO_INCREMENT PRIMARY KEY,
	change_ref VARCHAR(200),
	change_title VARCHAR(200),
	change_description VARCHAR(1000),
	created_by VARCHAR(100),
	assigned_to VARCHAR(100),
	type_id INT,
	priority_id INT,
	status_id INT, 
	estimated_impact VARCHAR(200),
	change_start_date DATE,
	change_end_date DATE,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	completed_at DATETIME NULL,
	CONSTRAINT fk_cr_user_created FOREIGN KEY (created_by) REFERENCES users(user_id) ON DELETE RESTRICT,
	CONSTRAINT fk_cr_user_assigned FOREIGN KEY (assigned_to) REFERENCES users(user_id) ON DELETE SET NULL,
	CONSTRAINT fk_cr_type FOREIGN KEY (type_id) REFERENCES change_types(type_id) ON DELETE RESTRICT,
	CONSTRAINT fk_cr_priority FOREIGN KEY (priority_id) REFERENCES priorities(priority_id) ON DELETE RESTRICT,
	CONSTRAINT fk_cr_status FOREIGN KEY (status_id) REFERENCES change_statuses(status_id) ON DELETE RESTRICT
);

CREATE TABLE APPROVALS (
	approval_id BIGINT AUTO_INCREMENT PRIMARY KEY,
	change_id BIGINT,
	approvar_id INT,
	decision ENUM('Pending','Approved','Rejected') DEFAULT 'Pending',
	comments varchar(500),
	CONSTRAINT fk_approval_change FOREIGN KEY (change_id) REFERENCES change_request(change_id) ON DELETE CASCADE,
	CONSTRAINT fk_approval_user FOREIGN KEY (approver_id) REFERENCES users(user_id) ON DELETE RESTRICT
);

CREATE TABLE comments (
	comment_id BIGINT AUTO_INCREMENT PRIMARY KEY,
	change_id BIGINT,
	user_id INT NOT NULL,
	comment varchar(1000),
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT fk_comment_change FOREIGN KEY (change_id) REFERENCES change_request(change_id) ON DELETE CASCADE,
	CONSTRAINT fk_comment_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE RESTRICT
);


CREATE TABLE attachments (
	attachment_id BIGINT AUTO_INCREMENT PRIMARY KEY,
	change_id BIGINT,
	file_name VARCHAR(255),
	file_url TEXT,
	mime_type VARCHAR(100),
	uploaded_by INT,
	uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT fk_attach_change FOREIGN KEY (change_id) REFERENCES change_requests(change_id) ON DELETE CASCADE,
	CONSTRAINT fk_attach_user FOREIGN KEY (uploaded_by) REFERENCES users(user_id) ON DELETE SET NULL
);


-- audit log
CREATE TABLE audit_log (
	audit_id BIGINT AUTO_INCREMENT PRIMARY KEY,
	change_id BIGINT,
	action VARCHAR(200),
	performed_by INT,
	performed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	details JSON NULL,
	CONSTRAINT fk_audit_change FOREIGN KEY (change_id) REFERENCES change_requests(change_id) ON DELETE SET NULL
);


-- dashboard KPI mapping: map friendly KPI keys to SQL templates (used by chatbot)
CREATE TABLE dashboard_kpis (
	kpi_id INT AUTO_INCREMENT PRIMARY KEY,
	kpi_key VARCHAR(100) UNIQUE NOT NULL, -- e.g. total_changes, open_changes
	display_name VARCHAR(200) NOT NULL,
	description TEXT,
	sql_template TEXT, -- parameterized SQL text e.g. "SELECT COUNT(*) AS value FROM change_requests WHERE status_id = {status_id} AND created_at >= '{start_date}'"
	visualization_type VARCHAR(50),
	last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);


-- indexes
CREATE INDEX idx_change_created_at ON change_requests(created_at);
CREATE INDEX idx_change_status ON change_requests(status_id);
CREATE INDEX idx_change_type ON change_requests(type_id);
CREATE INDEX idx_change_priority ON change_requests(priority_id);



-- sample data (small set to begin)
INSERT INTO USERS (full_name, email, role) VALUES
('Aisha Al-Harthy','aisha@example.com','Change Manager'),
('Omar Saif','omar@example.com','Approver'),
('Laila Hassan','laila@example.com','Requester');


INSERT INTO CHANGE_TYPE (name, description) VALUES
('Infrastructure','Changes to servers/network'),
('Application','Code or config changes'),
('Process','Process changes');


INSERT INTO CHANGE_STATUS (name) VALUES
('Open'),('In Progress'),('Closed'),('Awaiting Approval');


INSERT INTO PRIORITIES (name, level) VALUES
('High',1),('Medium',2),('Low',3);


INSERT INTO change_request (change_ref, title, description, created_by, assigned_to, type_id, priority_id, status_id, estimated_impact, requested_start_date, requested_end_date)
VALUES
('CR-1001','Patch web server','Urgent security patch',3,1,1,1,1,'High','2025-09-01','2025-09-02'),
('CR-1002','Update login module','Authentication improvements',3,2,2,2,4,'Medium','2025-09-10','2025-09-12');


INSERT INTO APPROVALS (change_id, approver_id, decision, decision_date) VALUES
(1,2,'Approved','2025-09-01 10:00:00');


-- example KPI mapping
INSERT INTO dashboard_kpis (kpi_key, display_name, description, sql_template, visualization_type) VALUES
('total_changes','Total change requests','Total number of change requests','SELECT COUNT(*) AS value FROM change_requests WHERE 1=1','card'),
('open_changes','Open change requests','Number of changes with status Open','SELECT COUNT(*) AS value FROM change_requests cr JOIN change_statuses cs ON cr.status_id = cs.status_id WHERE cs.name = ''Open'' {context_filter}','card');