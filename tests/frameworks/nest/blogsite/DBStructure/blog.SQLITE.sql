CREATE TABLE tBlogSections  ( 
	IDSection	INTEGER PRIMARY key,
	Title    	TEXT NOT NULL CONSTRAINT PK_BlogTag UNIQUE ON CONFLICT ABORT
);

CREATE TABLE tBlogData  ( 
	IDBLOG   	INTEGER PRIMARY key,
	Title    	TEXT NOT NULL,
	Content  	TEXT NOT NULL,
	Date     	NUMERIC NOT NULL DEFAULT CURRENT_TIMESTAMP,
	Author   	TEXT NOT NULL,
	IDSection	INTEGER UNSIGNED NOT NULL,
	FOREIGN KEY(IDSection) REFERENCES tBlogSections(IDSection)
);

CREATE TABLE tBlogComments  ( 
	IDBLOGCOMMENTS	INTEGER PRIMARY key,
	Content       	TEXT NOT NULL,
	Author        	TEXT NOT NULL,
	email         	TEXT NULL,
	website       	TEXT NULL,
	Date          	NUMERIC NOT NULL DEFAULT CURRENT_TIMESTAMP,
	IDBLOG        	INTEGER UNSIGNED NOT NULL,
	FOREIGN KEY(IDBLOG) REFERENCES tBlogData(IDBLOG)
);



CREATE TABLE tBlogTags  ( 
	IDBLOG	INTEGER NOT NULL,
	TAG   	TEXT NOT NULL,
	CONSTRAINT PK_BlogTag UNIQUE (IDBLOG, TAG),
	FOREIGN KEY(IDBLOG) REFERENCES tBlogData(IDBLOG)
);

CREATE INDEX FK_tBlogComments_tBlog ON tBlogComments(IDBLOG);
CREATE INDEX IDX_BlogComments_date 	ON tBlogComments(Date);
CREATE INDEX FK_tBlogData_BlogSection  ON tBlogData(IDSection);
CREATE INDEX IDX_blogdata_author 	ON tBlogData(Author);
CREATE INDEX IDX_blogdata_date 	ON tBlogData(Date);
CREATE INDEX FK_blogtag_blogdata 	ON tBlogTags(IDBLOG);