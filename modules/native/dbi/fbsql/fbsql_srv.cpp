/*
 * FALCON - The Falcon Programming Language.
 * FILE: fbsql_srv.cpp
 *
 * Firebird Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Mon, 06 Dec 2010 12:10:39 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>
#include <stdio.h>

#include <time.h>

#include <falcon/engine.h>
#include <falcon/dbi_error.h>
#include "fbsql_mod.h"

namespace Falcon
{

/******************************************************************************
 * Main service class
 *****************************************************************************/

void DBIServiceFB::init()
{
}

static void checkParamNumber(char *&dpb, const String& value, byte dpb_type, const String &option )
{
   if ( value.length() )
   {
      int64 res;
      if( ! value.parseInt(res) )
         throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
                  .extra( option + "=" + value )
               );

      *dpb++ = dpb_type;
      *dpb++ = 1;
      *dpb++ = (byte) res;
   }
}

static void checkParamYesOrNo(char *&dpb, const String& value, byte dpb_type, const String &option )
{
   if ( value.size() )
   {
      *dpb++ = dpb_type;
      *dpb++ = 1;

     if( value.compareIgnoreCase( "yes" ) == 0 )
        *dpb++ = (byte) 1;
     else if( value.compareIgnoreCase( "no" ) == 0 )
        *dpb++ = (byte) 0;
     else
        throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
                 .extra( option + "=" + value )
              );
   }
}

static void checkParamString(char *&dpb, const String& value, const char* szValue, byte dpb_type )
{
   if ( value.size() )
   {
       *dpb = dpb_type;
       ++dpb;
       *dpb = (char) value.size();
       ++dpb;
       strcpy( dpb, szValue );
       dpb += value.size();
   }
}


DBIHandle *DBIServiceFB::connect( const String &parameters )
{
   isc_db_handle handle = 0L;

   char dpb_buffer[256*10], *dpb, *p;
   // User name (uid)
   // Password (pwd)

   dpb = dpb_buffer;
   int dpb_length;

   // Parse the connection string.
   DBIConnParams connParams;

   // add Firebird specific parameters
   // Encrypted password (epwd)
   String sPwdEnc; const char* szPwdEncode;
   connParams.addParameter( "epwd", sPwdEnc, &szPwdEncode );

   // Role name (role)
   String sRole; const char* szRole;
   connParams.addParameter( "role", sRole, &szRole );

   // System database administrator’s user name (sa)
   String sSAName; const char* szSAName;
   connParams.addParameter( "sa", sSAName, &szSAName );

   // Authorization key for a software license (license)
   String sLicense; const char* szLicense;
   connParams.addParameter( "license", sLicense, &szLicense );

   // Database encryption key (ekey)
   String sKey; const char* szKey;
   connParams.addParameter( "ekey", sKey, &szKey );

   // Number of cache buffers (nbuf)
   String sNBuf;
   connParams.addParameter( "nbuf", sNBuf );

   // dbkey context scope (kscope)
   String sDBKeyScope;
   connParams.addParameter( "kscope", sDBKeyScope );

   // Specify whether or not to reserve a small amount of space on each database
   // --- page for holding backup versions of records when modifications are made (noreserve)
   String sNoRserve;
   connParams.addParameter( "reserve", sNoRserve );

   // Specify whether or not the database should be marked as damaged (dmg)
   String sDmg;
   connParams.addParameter( "dmg", sDmg );

   // Perform consistency checking of internal structures (verify)
   String sVerify;
   connParams.addParameter( "verify", sVerify );

   // Activate the database shadow, an optional, duplicate, in-sync copy of the database (shadow)
   String sShadow;
   connParams.addParameter( "shadow", sShadow );

   // Delete the database shadow (delshadow)
   String sDelShadow;
   connParams.addParameter( "delshadow", sDelShadow );

   // Activate a replay logging system to keep track of all database calls (beginlog)
   String sBeginLog;
   connParams.addParameter( "beginlog", sBeginLog );

   // Deactivate the replay logging system (quitlog)
   String sQuitLog;
   connParams.addParameter( "quitlog", sQuitLog );

   // Language-specific message file  (lcmsg)
   String sLcMsg; const char* szLcMsg;
   connParams.addParameter( "lcmsg", sLcMsg, &szLcMsg );

   // Character set to be utilized (lctype)
   String sLcType; const char* szLcType;
   connParams.addParameter( "lctype", sLcType, &szLcType );

   // Connection timeout (tout)
   String sTimeout;
   connParams.addParameter( "tout", sTimeout );

   if( ! connParams.parse( parameters ) )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
         .extra( parameters )
      );
   }

   // create the dpb; first the numerical values.
   *dpb++ = isc_dpb_version1;

   checkParamNumber( dpb, sNBuf, isc_dpb_num_buffers, "nbuf" );
   checkParamNumber( dpb, sTimeout, isc_dpb_connect_timeout, "tout" );

   checkParamYesOrNo( dpb, sDBKeyScope, isc_dpb_no_reserve, "kscope" );
   checkParamYesOrNo( dpb, sNoRserve, isc_dpb_no_reserve, "reserve" );
   checkParamYesOrNo( dpb, sDmg, isc_dpb_damaged, "dmg" );
   checkParamYesOrNo( dpb, sVerify, isc_dpb_verify, "verify" );
   checkParamYesOrNo( dpb, sShadow, isc_dpb_activate_shadow, "shadow" );
   checkParamYesOrNo( dpb, sDelShadow, isc_dpb_delete_shadow, "delshadow" );
   checkParamYesOrNo( dpb, sBeginLog, isc_dpb_begin_log, "beginlog" );
   checkParamYesOrNo( dpb, sQuitLog, isc_dpb_quit_log, "sQuitLog" );

   checkParamString( dpb, connParams.m_sUser, connParams.m_szUser, isc_dpb_user_name );
   checkParamString( dpb, connParams.m_sPassword, connParams.m_szPassword, isc_dpb_password );
   checkParamString( dpb, sPwdEnc, szPwdEncode, isc_dpb_password_enc );
   checkParamString( dpb, sRole, szRole, isc_dpb_sql_role_name );
   checkParamString( dpb, sLicense, szLicense, isc_dpb_license );
   checkParamString( dpb, sKey, szKey, isc_dpb_encrypt_key );
   //checkParamString( dpb, sLcMsg, szLcMsg, isc_dpb_lc_messages );
   // We'll ALWAYS use AutoCString to talk with Firebird, as such we'll ALWAYS use UTF8
   checkParamString( dpb, "UTF8", "UTF8", isc_dpb_lc_messages );

   /* Attach to the database. */
   ISC_STATUS status_vector[20];

   isc_attach_database(status_vector, strlen(connParams.m_szDb), connParams.m_szDb, &handle,
         dpb-dpb_buffer,
         dpb_buffer);

   if ( status_vector[0] == 1 && status_vector[1] )
   {
      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_CONNECT, status_vector );
   }

   return new DBIHandleFB( handle );
}


CoreObject *DBIServiceFB::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findWKI( "FirebirdSQL" );
   if ( cl == 0 || ! cl->isClass() )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_INVALID_DRIVER, __LINE__ ) );
   }

   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );

   return obj;
}

} /* namespace Falcon */

/* end of fbsql_srv.cpp */

