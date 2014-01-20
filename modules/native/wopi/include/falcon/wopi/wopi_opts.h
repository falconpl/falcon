/*
   FALCON - The Falcon Programming Language.
   FILE: wopi_opts.h

   Falcon Web Oriented Programming Interface.

   Options understood by the WOPI module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 25 Apr 2010 17:02:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

//==============================================================
// useful macros
//

#ifndef FALCON_MODULE_WOPI_OPTS_H
#define WOPI_OPT_SESSION_MODE_FILE "FILE"
#define WOPI_OPT_SESSION_MODE_SHMEM "NONE"
#define WOPI_OPT_SESSION_MODE_SHF  "SHF"
#define WOPI_OPT_SESSION_MODE_NONE "SHMEM"

#define WOPI_OPT_SESSION_MODE_FILE_ID  0
#define WOPI_OPT_SESSION_MODE_SHMEM_ID 1
#define WOPI_OPT_SESSION_MODE_SHF_ID   2
#define WOPI_OPT_SESSION_MODE_NONE_ID  3

#define WOPI_OPT_BOOL_ON_ID    1
#define WOPI_OPT_BOOL_OFF_ID   0
#define WOPI_OPT_BOOL_ON       "ON"
#define WOPI_OPT_BOOL_OFF      "OFF"

#define WOPI_OPT_LOG_MODE_SILENT_ID 0
#define WOPI_OPT_LOG_MODE_LOG_ID    1
#define WOPI_OPT_LOG_MODE_KIND_ID   2
#define WOPI_OPT_LOG_MODE_FULL_ID   3

#define WOPI_OPT_LOG_LEVEL_OFF      "OFF"
#define WOPI_OPT_LOG_LEVEL_CRIT     "CRIT"
#define WOPI_OPT_LOG_LEVEL_ERR      "ERR"
#define WOPI_OPT_LOG_LEVEL_WARN     "WARN"
#define WOPI_OPT_LOG_LEVEL_INFO     "INFO"
#define WOPI_OPT_LOG_LEVEL_DET      "DET"
#define WOPI_OPT_LOG_LEVEL_DBG      "DBG"
#define WOPI_OPT_LOG_LEVEL_DBG1     "DBG1"
#define WOPI_OPT_LOG_LEVEL_DBG2     "DBG2"

#define WOPI_OPT_LOG_LEVEL_OFF_ID   -1
#define WOPI_OPT_LOG_LEVEL_CRIT_ID  0
#define WOPI_OPT_LOG_LEVEL_ERR_ID   1
#define WOPI_OPT_LOG_LEVEL_WARN_ID  2
#define WOPI_OPT_LOG_LEVEL_INFO_ID  3
#define WOPI_OPT_LOG_LEVEL_DET_ID   4
#define WOPI_OPT_LOG_LEVEL_DBG_ID   5
#define WOPI_OPT_LOG_LEVEL_DBG1_ID  6
#define WOPI_OPT_LOG_LEVEL_DBG2_ID  7

#endif


//==============================================================
// Magic to create the options
//
#undef WOPI_OPTION
#undef WOPI_OPTION_DFLT
#undef WOPI_OPTION_CHECK

#ifdef WOPI_OPTION_REALIZE
#define WOPI_OPTION( __type__, __name__, __desc__ ) \
   addConfigOption( ::Falcon::WOPI::Wopi::ConfigEntry::e_t_##__type__, #__name__, __desc__ );

#define WOPI_OPTION_DFLT( __type__, __name__, __desc__, __dflt__ ) \
   addConfigOption( ::Falcon::WOPI::Wopi::ConfigEntry::e_t_##__type__, #__name__, __desc__, __dflt__ );

#define WOPI_OPTION_CHECK( __type__, __name__, __desc__, __dflt__ ) \
   addConfigOption( ::Falcon::WOPI::Wopi::ConfigEntry::e_t_##__type__, #__name__, __desc__, __dflt__, &s_check_wopi_opt_##__name__ );

#else

   #ifdef WOPI_OPTION_DECLARE
      #define WOPI_OPTION( __type__, __name__, __desc__ ) extern const char* OPT_##__name__;
      #define WOPI_OPTION_DFLT( __type__, __name__, __desc__, __dflt__ ) extern const char* OPT_##__name__;
      #define WOPI_OPTION_CHECK( __type__, __name__, __desc__, __dflt__ ) extern const char* OPT_##__name__;
   #else

      #ifdef WOPI_OPTION_DEFINE
         #define WOPI_OPTION( __type__, __name__, __desc__ ) const char* OPT_##__name__ = #__name__;
         #define WOPI_OPTION_DFLT( __type__, __name__, __desc__, __dflt__ ) const char* OPT_##__name__ = #__name__;
         #define WOPI_OPTION_CHECK( __type__, __name__, __desc__, __dflt__ ) const char* OPT_##__name__ = #__name__;
      #else
         #define WOPI_OPTION( __type__, __name__, __desc__ )
         #define WOPI_OPTION_DFLT( __type__, __name__, __desc__, __dflt__ )
         #define WOPI_OPTION_CHECK( __type__, __name__, __desc__, __dflt__ )
      #endif
   #endif
#endif

//==============================================================
// Options list
//

WOPI_OPTION      ( string, LoadPath, "Default load path for the Falcon engine" )
WOPI_OPTION_DFLT ( string, SourceEncoding, "Default source encoding for the Falcon engine", "utf8" )
WOPI_OPTION_DFLT ( string, OutputEncoding, "Default output encoding for the Falcon engine", "utf8" )
WOPI_OPTION      ( string, FalconHandler, "General script handling incoming requests" )
WOPI_OPTION_DFLT ( string, SessionField, "Session ID field name (defaluts to SID)", "SID" )
WOPI_OPTION_CHECK( int   , SessionMode, "Session storage mode - can be one of: NONE FILE SHMEM SHF", WOPI_OPT_SESSION_MODE_FILE )
WOPI_OPTION_DFLT ( int   , SessionTimeout, "Default session timeout in seconds", 600 )
WOPI_OPTION_CHECK( int   , SessionAuto, "Automatically apply session variables at script startup: ON or OFF", WOPI_OPT_BOOL_ON )
WOPI_OPTION_CHECK( int   , MaxUploadSize, "Maximum upload size in kilobytes", 2048 )
WOPI_OPTION_CHECK( int   , MaxMemoryUploadSize, "Upload size in kilobytes under which memory-only upload is used", 4 )

WOPI_OPTION_CHECK( int   , WebLogLevel, "Send engine and/or script logs at the end of the page. Can be [OFF], CRIT, ERR, WARN, INFO, DET, DBG, DBG1, DBG2", WOPI_OPT_LOG_LEVEL_OFF )
WOPI_OPTION_CHECK( int   , AppLogLevel, "Send engine and/or script logs to the application-specific log facility. Can be OFF, CRIT, ERR, WARN, INFO, DET, [DBG], DBG1, DBG2", WOPI_OPT_LOG_LEVEL_DBG )
WOPI_OPTION_CHECK( int   , WebLogInternal, "Web log records Falcon engine logs not coming from the scritps; can be ON or [OFF]", WOPI_OPT_BOOL_OFF )
WOPI_OPTION_CHECK( int   , AppLogInternal, "App log records Falcon engine logs not coming from the scripts; to the application-specific log facility. Can be ON or [OFF]", WOPI_OPT_BOOL_OFF )

WOPI_OPTION_CHECK( int, ErrorFancyReport, "Send a full document when reporting an error", WOPI_OPT_BOOL_ON )
WOPI_OPTION      ( string, ErrorTemplateDocument, "HTML full Document template for error reporting (overrides fancy erorr reporting)" )
WOPI_OPTION      ( string, ErrorTemplateSection, "HTML template used to report errors appended to ongoing script output" )
WOPI_OPTION      ( string, ErrorTemplateEngine, "HTML template used to report engine or web errors (as part of Document Template or Section Template)" )
WOPI_OPTION      ( string, ErrorTemplateScript, "HTML template used to report script errors (as part of Document Template or Section Template)" )

// System specific default settings.
#ifdef FALCON_SYSTEM_WIN
WOPI_OPTION_DFLT( string, TempDir, "Local file system directory for temporary files", "C:\\TEMP" )
WOPI_OPTION_DFLT( string, UploadDir, "Directory where uploaded files are placed", "C:\\TEMP" )
#else
WOPI_OPTION_DFLT( string, TempDir, "Local file system directory for temporary files", "/tmp" )
WOPI_OPTION_DFLT( string, UploadDir, "Directory where uploaded files are placed", "/tmp" )
#endif

