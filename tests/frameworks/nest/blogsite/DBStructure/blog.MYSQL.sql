/*
Script generato da Aqua Data Studio 9.0.15 in data lug-07-2011 11:06:43 PM
Database: CKDB
Schema: <Tutti gli schemi>
Oggetti: TABLE
*/


CREATE TABLE tBlogComments  ( 
	IDBLOGCOMMENTS	bigint(20) UNSIGNED AUTO_INCREMENT NOT NULL,
	Content       	text NOT NULL,
	Author        	varchar(50) NOT NULL,
	email         	varchar(50) NULL,
	website       	varchar(50) NULL,
	Date          	timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
	IDBLOG        	bigint(20) UNSIGNED NOT NULL,
	PRIMARY KEY(IDBLOGCOMMENTS)
);
CREATE TABLE tBlogData  ( 
	IDBLOG   	bigint(20) UNSIGNED AUTO_INCREMENT NOT NULL,
	Title    	varchar(100) NOT NULL,
	Content  	text NOT NULL,
	Date     	timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
	Author   	varchar(50) NOT NULL,
	IDSection	int(10) UNSIGNED NOT NULL,
	PRIMARY KEY(IDBLOG)
);
CREATE TABLE tBlogSections  ( 
	IDSection	int(10) UNSIGNED AUTO_INCREMENT NOT NULL,
	Title    	varchar(50) NOT NULL,
	PRIMARY KEY(IDSection)
);
CREATE TABLE tBlogTags  ( 
	IDBLOG	bigint(20) UNSIGNED NOT NULL,
	TAG   	varchar(50) NOT NULL 
	);

CREATE INDEX FK_tBlogComments_tBlog USING BTREE 
	ON tBlogComments(IDBLOG);
CREATE INDEX IDX_BlogComments_date USING BTREE 
	ON tBlogComments(Date);
CREATE INDEX FK_tBlogData_BlogSection USING BTREE 
	ON tBlogData(IDSection);
CREATE INDEX IDX_blogdata_author USING BTREE 
	ON tBlogData(Author);
CREATE INDEX IDX_blogdata_date USING BTREE 
	ON tBlogData(Date);
CREATE INDEX FK_blogtag_blogdata USING BTREE 
	ON tBlogTags(IDBLOG);
ALTER TABLE tBlogTags
	ADD CONSTRAINT PK_BlogTag
	UNIQUE (IDBLOG, TAG);
ALTER TABLE tBlogComments
	ADD CONSTRAINT FK_tBlogComments_tBlog
	FOREIGN KEY(IDBLOG)
	REFERENCES tBlogData(IDBLOG)
	ON DELETE NO ACTION 
	ON UPDATE NO ACTION ;
ALTER TABLE tBlogData
	ADD CONSTRAINT FK_tBlogData_BlogSection
	FOREIGN KEY(IDSection)
	REFERENCES tBlogSections(IDSection)
	ON DELETE NO ACTION 
	ON UPDATE NO ACTION ;
ALTER TABLE tBlogTags
	ADD CONSTRAINT FK_blogtag_blogdata
	FOREIGN KEY(IDBLOG)
	REFERENCES tBlogData(IDBLOG)
	ON DELETE NO ACTION 
	ON UPDATE NO ACTION ;
