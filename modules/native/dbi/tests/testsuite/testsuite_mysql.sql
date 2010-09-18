/*******************************************************
* Falcon DBI - testsuite
*
*
* Script creating the testsuite database and the faltest
* user in the MySQL database for the DBI testsuite.
*
* Run with
*    $ mysql -u root -p < testsuite_mysql.sql
*
*/

create database IF NOT EXISTS testsuite
   character set utf8;

use testsuite;

DROP PROCEDURE IF EXISTS add_user_if_not_exist;
DELIMITER $$
CREATE PROCEDURE add_user_if_not_exist()
BEGIN
  DECLARE foo BIGINT DEFAULT 0 ;
  SELECT COUNT(*)
  INTO foo
    FROM `mysql`.`user`
      WHERE `User` = 'faltest' ;

  IF foo = 0 THEN
     CREATE USER faltest IDENTIFIED BY 'faltest';
     GRANT all on * to faltest;
  END IF;
END ;$$
DELIMITER ;

CALL add_user_if_not_exist() ;

DROP PROCEDURE IF EXISTS add_user_if_not_exist;






