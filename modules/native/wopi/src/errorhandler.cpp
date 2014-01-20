/*
   FALCON - The Falcon Programming Language.
   FILE: errorhandler.h

   Web Oriented Programming Interface.

   Generic base class to produce error reports on web.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 15 Jan 2014 12:36:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/wopi/src/errorhandler.cpp"

#include <falcon/string.h>
#include <falcon/error.h>
#include <falcon/log.h>
#include <falcon/wopi/client.h>
#include <falcon/wopi/errorhandler.h>
#include <falcon/wopi/wopi.h>

namespace Falcon {
namespace WOPI {

ErrorHandler::ErrorHandler(Log* log, bool bFancy):
   m_log(log)
{
   if(bFancy) {
      loadFancyErrorStyle();
   }
}

ErrorHandler::ErrorHandler( bool bFancy):
   m_log(0)
{
   if(bFancy) {loadFancyErrorStyle();}
}


ErrorHandler::~ErrorHandler()
{}



void ErrorHandler::renderError( Client* client, Error* e )
{
   String sTemp;
   if ( m_log != 0 )
   {
      String head;
      m_log->log( Log::fac_app, Log::lvl_error, "WOPI system error: " + e->heading(head) );
   }

   logError(client, e);

   String sDesc;
   e->describeTo(sDesc, true);

   String sError;
   if( ! m_scriptErrorTpl.empty() )
   {
      String fullCode, location, trace, subs;
      m_scriptErrorTpl.replace( FALCON_WOPI_ERRORTPL_CODE, e->fullCode(fullCode), sError );

      sError.replace( FALCON_WOPI_ERRORTPL_CLASS, e->className(), sTemp );
      sError = sTemp;
      if( ! e->errorDescription().empty() )
      {
         sError.replace( FALCON_WOPI_ERRORTPL_MESSAGE, e->errorDescription(), sTemp );
      }
      else
      {
         sError.replace( FALCON_WOPI_ERRORTPL_MESSAGE, e->describeErrorCode(e->errorCode()), sTemp );
      }
      sError = sTemp;
      sError.replace( FALCON_WOPI_ERRORTPL_EXTRA, e->extraDescription(), sTemp );
      sError = sTemp;
      sError.replace( FALCON_WOPI_ERRORTPL_LOCATION, e->location(location), sTemp );
      sError = sTemp;
      sError.replace( FALCON_WOPI_ERRORTPL_SIGN, e->signature(), sTemp );
      sError = sTemp;
      sError.replace( FALCON_WOPI_ERRORTPL_TRACE, e->describeTrace(trace), sTemp );
      sError = sTemp;
      sError.replace( FALCON_WOPI_ERRORTPL_CAUSE, e->describeSubErrors(subs), sTemp );
      sError = sTemp;
      sError.replace( FALCON_WOPI_ERRORTPL_FULLDESC, sDesc, sTemp );
      sError = sTemp;

      if( e->systemError() != 0 )
      {
         String sSys;
         Falcon::Sys::_describeError( e->systemError(), sSys);
         if( sSys.empty() )
         {
            sSys.N(e->systemError() );
         }
         else {
            sSys = String( "" ).N(e->systemError()).A(": ") + sSys;
         }

         sError.replace( FALCON_WOPI_ERRORTPL_SYS, sSys, sTemp, 1 );
         sError = sTemp;
      }
      else
      {
         sError.replace( FALCON_WOPI_ERRORTPL_SYS, "", sTemp, 1 );
         sError = sTemp;
      }

      if( e->hasRaised() )
      {
         sError.replace( FALCON_WOPI_ERRORTPL_RAISED, e->raised().describe(3,1024), sTemp );
         sError = sTemp;
      }
      else {
         sError.replace( FALCON_WOPI_ERRORTPL_RAISED, "", sTemp );
         sError = sTemp;
      }
   }
   else
   {
      sError = sDesc;
   }

   String sRender;
   if ( client->reply()->isCommited() )
   {
      if( m_errorSectionTpl.empty() )
      {
         sRender = sError;
      }
      else
      {
         sRender = m_errorSectionTpl;
      }
   }
   else
   {
      client->reply()->status(500);
      client->reply()->reason(errorDesc(500));

      if( m_errorDocTpl.empty() )
      {
         sRender = sError;
         client->reply()->setContentType("text", "plain", "utf-8");
      }
      else
      {
         client->reply()->setContentType("text", "html", "utf-8");
         sRender = m_errorDocTpl;
         sRender.replace( FALCON_WOPI_ERRORTPL_TITLE_BAR, String("Error 500"), sTemp, 1 );
         sRender = sTemp;
      }
   }

   sRender.replace( FALCON_WOPI_ERRORTPL_TITLE, "Error in " + client->request()->getUri(), sTemp );
   sRender = sTemp;
   sRender.replace( FALCON_WOPI_ERRORTPL_REPORT, sError, sTemp );
   sRender = sTemp;
   sRender.replace( FALCON_WOPI_ERRORTPL_VERSION, Engine::instance()->version(), sTemp );
   sRender = sTemp;
   replyError(client, sRender);
}


void ErrorHandler::renderSysError( Client* client, int code, const String& message )
{
   String sError,sTemp;
   sError.N(code);
   sError += " ";

   if( ! message.empty() )
   {
      sError += message;
   }
   else {
      sError += errorDesc(code);
   }

   if ( m_log != 0 )
   {
      m_log->log( Log::fac_app, Log::lvl_error, "WOPI system error: " + sError );
   }

   logSysError(client, code, message);

   if( ! m_engineErrorTpl.empty() )
   {
      sError = "";
      m_engineErrorTpl.replace( FALCON_WOPI_ERRORTPL_CODE, String("").N(code), sError );
      sError.replace( FALCON_WOPI_ERRORTPL_EXTRA, message, sTemp );
      sError = sTemp;
      sError.replace( FALCON_WOPI_ERRORTPL_MESSAGE, errorDesc(code), sTemp );
      sError = sTemp;
   }
   else
   {
      // keep the raw sError we have
   }

   String sRender;
   if ( client->reply()->isCommited() )
   {
      if( m_errorSectionTpl.empty() )
      {
         sRender = sError;
      }
      else
      {
         sRender = m_errorSectionTpl;
      }
   }
   else
   {
      client->reply()->status(code);
      client->reply()->reason(errorDesc(code));

      if( m_errorDocTpl.empty() )
      {
         sRender = sError;
         client->reply()->setContentType("text", "plain", "utf-8");
      }
      else
      {
         sRender = m_errorDocTpl;
         client->reply()->setContentType("text", "html", "utf-8");
         sRender.replace( FALCON_WOPI_ERRORTPL_TITLE_BAR, String("Error ").N(code), sTemp );
         sRender = sTemp;
      }
   }

   sRender.replace( FALCON_WOPI_ERRORTPL_TITLE, String("System Error ").N(code), sTemp );
   sRender = sTemp;
   sRender.replace( FALCON_WOPI_ERRORTPL_REPORT, sError, sTemp );
   sRender = sTemp;
   sRender.replace( FALCON_WOPI_ERRORTPL_VERSION, Engine::instance()->version(), sTemp );
   sRender = sTemp;
   replyError(client, sRender);
}


/** Returns the HTTP error code default description. */
String ErrorHandler::defaultErrorDesc( int errorID )
{
   switch( errorID )
   {
   // continue codes
   case 100: return "Continue";
   case 101: return "Switching Protocols";

   // Success codes
   case 200: return "OK";
   case 201: return "Created";
   case 202: return "Accepted";
   case 203: return "Non-Authoritative Information";
   case 204: return "No Content";
   case 205: return "Reset Content";
   case 206: return "Partial Content";

   // Redirection Codes
   case 300: return "Multiple Choices";
   case 301: return "Moved Permanently";
   case 302: return "Found";
   case 303: return "See Other";
   case 304: return "Not Modified";
   case 305: return "Use Proxy";
   case 307: return "Temporary Redirect";

   // Client Error Codes

   case 400: return "Bad Request";
   case 401: return "Unauthorized";
   case 402: return "Payment Required";
   case 403: return "Forbidden";
   case 404: return "Not Found";
   case 405: return "Method Not Allowed";
   case 406: return "Not Acceptable";
   case 407: return "Proxy Authentication Required";
   case 408: return "Request Timeout";
   case 409: return "Conflict";
   case 410: return "Gone";
   case 411: return "Length Required";
   case 412: return "Precondition Failed";
   case 413: return "Request Entity Too Large";
   case 414: return "Request-URI Too Large";
   case 415: return "Unsupported Media Type";
   case 416: return "Requested Range Not Satisfiable";
   case 417: return "Expectation Failed";

   // Server Error Codes
   case 500: return "Internal Server Error";
   case 501: return "Not Implemented";
   case 502: return "Bad Gateway";
   case 503: return "Service Unavailable";
   case 504: return "Gateway Timeout";
   case 505: return "HTTP Version not supported";

   }

   return "Unknown code";
}


String ErrorHandler::errorDesc( int code )
{
   return ErrorHandler::defaultErrorDesc(code);
}



void ErrorHandler::setErrorDocumentTemplate( const String& tpl )
{
   m_errorDocTpl = tpl;
}


void ErrorHandler::setErrorSectionTemplate( const String& tpl )
{
   m_errorSectionTpl = tpl;
}


void ErrorHandler::setEngineErrorTemplate( const String& tpl )
{
   m_engineErrorTpl = tpl;
}


void ErrorHandler::setScriptErrorTemplate( const String& tpl )
{
   m_scriptErrorTpl = tpl;
}


void ErrorHandler::loadFancyErrorStyle()
{
   loadFancyDocumentTpl();
   loadFancySectionTpl();
   loadFancyScriptErrorTpl();
   loadFancyEngineErrorTpl();
}


void ErrorHandler::loadCrudeErrorStyle()
{
   m_errorDocTpl = "";
   m_errorSectionTpl = "";
   m_scriptErrorTpl = "";
   m_engineErrorTpl = "";
}

void ErrorHandler::loadErrorDocumentTemplate( const String& localFile )
{
   try {
      m_errorDocTpl = "";
      loadErrorDocument( localFile, m_errorDocTpl );
   }
   catch(Error* e)
   {
      loadFancyDocumentTpl(e->describe());
      throw e;
   }
}

void ErrorHandler::loadErrorSectionTemplate( const String& localFile )
{
   try {
      m_errorSectionTpl = "";
      loadErrorDocument( localFile, m_errorSectionTpl );
   }
   catch(Error* e)
   {
      loadFancySectionTpl(e->describe());
      throw e;
   }
}


void ErrorHandler::loadEngineErrorTemplate( const String& localFile )
{
   try {
      m_engineErrorTpl = "";
      loadErrorDocument( localFile, m_engineErrorTpl );
   }
   catch(Error* e)
   {
      loadFancyEngineErrorTpl(e->describe());
      throw e;
   }
}



void ErrorHandler::loadScriptErrorTemplate( const String& localFile )
{
   try {
      m_scriptErrorTpl = "";
      loadErrorDocument( localFile, m_scriptErrorTpl );
   }
   catch(Error* e)
   {
      loadFancyScriptErrorTpl(e->describe());
      throw e;
   }
}


void ErrorHandler::loadConfigFromWopi( Wopi* wopi )
{
   String error;
   int64 iValue;
   wopi->getConfigValue( OPT_ErrorFancyReport, iValue, error );

   if( iValue == WOPI_OPT_BOOL_ON_ID )
   {
      loadFancyErrorStyle();
   }
   else
   {
      loadCrudeErrorStyle();
   }

   String sDoc, sEng, sScript, sSection;
   wopi->getConfigValue( OPT_ErrorTemplateDocument, sDoc, error );
   wopi->getConfigValue( OPT_ErrorTemplateEngine, sEng, error );
   wopi->getConfigValue( OPT_ErrorTemplateScript, sScript, error );
   wopi->getConfigValue( OPT_ErrorTemplateSection, sSection, error );

   if( ! sDoc.empty() )
   {
      loadErrorDocumentTemplate(sDoc);
   }

   if( ! sEng.empty() )
   {
      loadEngineErrorTemplate(sEng);
   }

   if( ! sScript.empty() )
   {
      loadScriptErrorTemplate(sScript);
   }

   if( ! sSection.empty() )
   {
      loadErrorSectionTemplate(sSection);
   }
}


void ErrorHandler::loadFancyDocumentTpl( const String& problem )
{
   String temp =
      "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">\n"
      "<html>\n"
      "<head>\n"
      "<title>Falcon WOPI - " FALCON_WOPI_ERRORTPL_TITLE_BAR "</title>\n"
      "</head>\n"
      "<body>\n"
      "<div style=\"text-align: center; color: #A55\">"
               "<div style=\"display: inline-block; vertical-align:middle\"><a href=\"http://www.falconpl.org\"><img border=\"0\" src=\"http://www.falconpl.org/images/logo.png\"></a></div>"
               "<div style=\"font: italic bold 24pt arial, sans-serif; display: inline-block; vertical-align:middle\">&nbsp; <a href=\"http://www.falconpl.org\">The Falcon Programming Language</a></div>"
               "<div style=\"font: bold 16pt arial, sans-serif; color: #005;\">Web Oriented Programming Interface</div>"
               "<div style=\"font: italic 10pt arial, sans-serif; color: #000;\">%VERSION-INFO%</div>"
      "</div>"
      "<div>"
      "<h1>" FALCON_WOPI_ERRORTPL_TITLE "</h1>"
      FALCON_WOPI_ERRORTPL_REPORT
      "</div>"
      "%PROBLEM-REPORT%\n"
      "<hr/><div style=\"font: italic 10pt 'Times new roman', serif;\">The Falcon Programming Language %VERSION-INFO% - WOPI standard error report form.</div>"
      "</body>\n"
      "</html>\n";

   setProblem( problem, temp, m_errorDocTpl );
}


void ErrorHandler::setProblem( const String& problem, const String& source, String& target )
{
   if( problem.empty() )
   {
      source.replace("%PROBLEM-REPORT%", "", target);
   }
   else {
      source.replace("%PROBLEM-REPORT%",
      "<hr/>\n<div style=\"font: 12pt Arial, Verdana, sans-serif; color: #844;\">"
      "Additionally, the required error document template file couldn't be loaded for the following reason:<br/><pre>"
          +problem + "</pre>\n</div>",
          target );
   }
}


void ErrorHandler::loadFancySectionTpl( const String& problem )
{
   String temp =
      "<div style=\"border: black solid 1px; margin:5px; padding: 5px \">\n"
      "<div style=\"font: bold 16pt Arial, Verdana, Sans;\">Falcon WOPI - "
      FALCON_WOPI_ERRORTPL_TITLE "</div>\n"
      "<div style=\"font: 12pt Arial, Verdana, Sans;\">"
      FALCON_WOPI_ERRORTPL_REPORT
      "</div>"
      "%PROBLEM-REPORT%\n"
      "<hr/><div style=\"font: italic 10pt 'Times new roman', serif;\">The Falcon Programming Language %VERSION-INFO% - WOPI inline error report form.</div>"
      "</div>";
   setProblem( problem, temp, m_errorSectionTpl );
}


void ErrorHandler::loadFancyEngineErrorTpl( const String& problem )
{
   String temp =
            "<div style=\"font: 14pt 'Times new roman', serif;\">"
            "<p>Your request has generated an error of type <b>" FALCON_WOPI_ERRORTPL_CODE "</b> (" FALCON_WOPI_ERRORTPL_MESSAGE ").</p>"
            "<p>" FALCON_WOPI_ERRORTPL_EXTRA "</p>"
            "</div>"
            "%PROBLEM-REPORT%\n";
   setProblem( problem, temp, m_engineErrorTpl );
}


void ErrorHandler::loadFancyScriptErrorTpl( const String& problem )
{
   String temp =
            "<div style=\"font: arial, sans-serif 11pt;\">"
            "<table>"
            "<tbody>"
            "<tr><td colspan=\"2\"><b> Error " FALCON_WOPI_ERRORTPL_CODE "</b>&nbsp;-&nbsp;" FALCON_WOPI_ERRORTPL_MESSAGE "</td></tr>"
            "<tr><td>Extended info:</td><td>" FALCON_WOPI_ERRORTPL_EXTRA "</td></tr>"
            "<tr><td>Class:</td><td>" FALCON_WOPI_ERRORTPL_CLASS "</td></tr>"
            "<tr><td>Error location:</td><td>" FALCON_WOPI_ERRORTPL_LOCATION "</td></tr>"
            "<tr><td>System-level error:</td><td>" FALCON_WOPI_ERRORTPL_SYS "</td></tr>"
            "<tr><td>Signed by:</td><td>" FALCON_WOPI_ERRORTPL_SIGN "</td></tr>"
            "<tr><td>Raised value:</td><td><pre>" FALCON_WOPI_ERRORTPL_RAISED "</pre></td></tr>"
            "<tr><td style=\"vertica-align:top\">Trace:</td><td><pre>" FALCON_WOPI_ERRORTPL_TRACE "</pre></td></tr>"
            "<tr><td style=\"vertica-align:top\">Caused by:</td><td><pre>" FALCON_WOPI_ERRORTPL_CAUSE "</pre></td></tr>"
            "</tbody>\n</table>\n"
            "</div>"
            "%PROBLEM-REPORT%\n";
   setProblem( problem, temp, m_scriptErrorTpl );
}


void ErrorHandler::loadErrorDocument( const String& localFile, String& target )
{
   Transcoder* tc = Engine::instance()->getTranscoder("utf8");
   fassert( tc != 0);

   Stream* stream = Engine::instance()->vfs().openRO( localFile );
   TextReader tr(stream, tc);
   stream->decref();

   tr.readEof(target);
}

}
}

/* end of errortemplate.cpp */

