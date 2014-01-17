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

#ifndef _FALCON_WOPI_ERRORHANDLER_H_
#define _FALCON_WOPI_ERRORHANDLER_H_

#include <falcon/setup.h>
#include <falcon/types.h>

#define FALCON_WOPI_ERRORTPL_TITLE_BAR "%ERROR-TITLE-BAR%"
#define FALCON_WOPI_ERRORTPL_TITLE     "%ERROR-TITLE%"
#define FALCON_WOPI_ERRORTPL_REPORT    "%ERROR-REPORT%"
#define FALCON_WOPI_ERRORTPL_VERSION   "%VERSION-INFO%"

#define FALCON_WOPI_ERRORTPL_CODE      "%ERROR-CODE%"
#define FALCON_WOPI_ERRORTPL_CLASS     "%ERROR-CLASS%"
#define FALCON_WOPI_ERRORTPL_MESSAGE   "%ERROR-MESSAGE%"
#define FALCON_WOPI_ERRORTPL_NAME      "%ERROR-NAME%"
#define FALCON_WOPI_ERRORTPL_EXTRA     "%ERROR-EXTRA%"
#define FALCON_WOPI_ERRORTPL_LOCATION  "%ERROR-LOCATION%"
#define FALCON_WOPI_ERRORTPL_SIGN      "%ERROR-SIGNATURE%"
#define FALCON_WOPI_ERRORTPL_RAISED    "%ERROR-RAISED%"
#define FALCON_WOPI_ERRORTPL_TRACE     "%ERROR-TRACE%"
#define FALCON_WOPI_ERRORTPL_CAUSE     "%ERROR-CAUSE%"
#define FALCON_WOPI_ERRORTPL_SYS       "%ERROR-SYS%"

#define FALCON_WOPI_ERRORTPL_FULLDESC  "%ERROR-FULLDESC%"

namespace Falcon {
class String;
class Log;
class Error;

namespace WOPI {

class Client;
class Wopi;

/** Generic base class to produce error reports on web.
*
* \note The implementation is not thread-safe; be sure to
* set the error templates from thread-safe code.
*/
class ErrorHandler
{
public:
   /** Creates the error handler without a backup logger facility. */
   ErrorHandler( bool bFancy = true );

   /** Creates the error handler with a backup logger facility. */
   ErrorHandler( Log* log, bool bFancy = true );

   virtual ~ErrorHandler();

   /** Sends the error to the reply.
    * \param client The client that shall receive the error reporting.
    * \param message Pre-rendered message to be transmitted as-is.
    *
    * Subclasses must implement this method to provide correct output
    * for the correct
    */
   virtual void replyError( Client* client, const String& message ) = 0;

   /** Sends a WEB error to the logging facility.
    * \param client The client that caused the error.
    * \param code The HTTP system error code
    * \param message The message to be sent to the error logging facility as-is.
    *
    * Subclasses must implement this if they want to bypass
    * the standard Falcon logging system and immediately generate
    * an output on the error log reporting facility.
    *
    * A Falcon system error level log entry is generated immediately before
    * calling this method; this means that, if the WOPI implementation uses
    * the falcon logging system to generate log entries, the subclass needs
    * to implement is method as an empty (do-nothing) operation.
    *
    */
   virtual void logSysError( Client* client, int code, const String& message ) = 0;

   /** Sends a Falcon error description to the logging facility.
    * \param client The client that caused the error.
    * \param code The HTTP system error code
    * \param message The message to be sent to the error logging facility as-is.
    *
    * Subclasses must implement this if they want to bypass
    * the standard Falcon logging system and immediately generate
    * an output on the error log reporting facility.
    *
    * A Falcon system error level log entry is generated immediately before
    * calling this method; this means that, if the WOPI implementation uses
    * the falcon logging system to generate log entries, the subclass needs
    * to implement is method as an empty (do-nothing) operation.
    *
    */
   virtual void logError( Client* client, Error* error ) = 0;

   /** Generate a script error report.
    *
    * This is invoked by the handler users when a fatal script error is detected
    * (i.e. when the script terminates raising an error).
    */
   void renderError( Client* client, Error* e );

   /** Generate a system error report.
    *
    * This is invoked by the handler users when a HTTP level error is detected.
    * For instance, when an invalid document or script is being loaded, or in
    * case of redirect (temporary or permanent) requests being issued by the
    * page-to-file resolver.
    */
   void renderSysError(  Client* client, int code, const String& message="" );

   /** Returns the HTTP error code default description.
    *
    * The base version invokes defaultErrorDesc().
    * Virtualized to allow overriding.
    */
   virtual String errorDesc( int code );

   /** Returns the HTTP error code default description.
    *
    */
   static String defaultErrorDesc( int code );

   /** Loads the text of the error document.
    *
    * If the error document template is not empty, the reply type will be set to
    * text/html.
    */
   void setErrorDocumentTemplate( const String& tpl );

   /** Sets the text of the error section.
    *
    * The error section template is used when the engine detects a script-error
    * after the script has generated part of the output; in such case, generating
    * a whole document template is useless, while it might be useful to popup
    * an error message with prominent style mid-page.
    *
    */
   void setErrorSectionTemplate( const String& tpl );

   /** Sets the template for engine error reporting.
    *
    * If not set, the output will be [Numeric code] [Text explanation].
    */
   void setEngineErrorTemplate( const String& tpl );

   /** Sets the template for script error reporting.
    *
    * If not set, the output will be the raw result of Falcon::Error::describe().
    */
   void setScriptErrorTemplate( const String& tpl );


   /** Clears all the error reporting templates.
    *
    * This effectively turns error reporting into a simple text oriented output of the
    * crude error message.
    */
   void loadCrudeErrorStyle();

   /** Sets the default error handler document, engine error and script error template to a fancy report.
    *
    * This sets the default error reporting template to an internally generated template that presents
    * a minimally informative and graphically appealing form.
    */
   void loadFancyErrorStyle();

   /** Loads the text of an error document template from disk.
    *
    * If the error document template is not empty, the reply type will be set to
    * text/html.
    *
    * \note In case of I/O error while reading the given local file,
    * a special template reporting this fact is set, and the error is re-thrown.
    *
    * \note The template file is to be encoded in utf-8 format.
    *
    */
   void loadErrorDocumentTemplate( const String& localFile );

   /** Loads the text of an error section template from disk.
    *
    * The error section template is used when the engine detects a script-error
    * after the script has generated part of the output; in such case, generating
    * a whole document template is useless, while it might be useful to popup
    * an error message with prominent style mid-page.
    *
    * \note In case of I/O error while reading the given local file,
    * a special template reporting this fact is set, and the error is re-thrown.
    *
    * \note The template file is to be encoded in utf-8 format.
    */
   void loadErrorSectionTemplate( const String& localFile );

   /** Loads the text of an engine error reporting.
    *
    * \note In case of I/O error while reading the given local file,
    * a special template reporting this fact is set, and the error is re-thrown.
    *
    * \note The template file is to be encoded in utf-8 format.
    */
   void loadEngineErrorTemplate( const String& localFile );

   /** Loads the text of a script error reporting.
    *
    * \note In case of I/O error while reading the given local file,
    * a special template reporting this fact is set, and the error is re-thrown.
    *
    * \note The template file is to be encoded in utf-8 format.
    */
   void loadScriptErrorTemplate( const String& localFile );

   void loadConfigFromWopi( Wopi* wopi );

private:
   String m_errorDocTpl;
   String m_errorSectionTpl;
   String m_engineErrorTpl;
   String m_scriptErrorTpl;

   Log* m_log;

   void loadErrorDocument( const String& localFile, String& target );

   void loadFancyDocumentTpl( const String& problem = "" );
   void loadFancySectionTpl( const String& problem = "" );
   void loadFancyEngineErrorTpl( const String& problem = "" );
   void loadFancyScriptErrorTpl( const String& problem = "" );
   void setProblem( const String& problem, const String& source, String& target );
};

}
}

#endif /* _FALCON_WOPI_ERRORHANDLER_H_ */

/* errorhandler.h */
