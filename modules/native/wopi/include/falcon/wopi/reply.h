/*
   FALCON - The Falcon Programming Language.
   FILE: reply.h

   Web Oriented Programming Interface

   Object encapsulating reply.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 18 Feb 2010 14:22:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_WOPI_REPLY
#define FALCON_WOPI_REPLY

#include <falcon/wopi/utils.h>
#include <falcon/timestamp.h>

#define FALCON_WOPI_DEFAULT_REPLY_STATUS  200
#define FALCON_WOPI_DEFAULT_REPLY_REASON  "Ok"

namespace Falcon {
namespace WOPI {


class CookieParams
{
public:
   CookieParams();

   CookieParams& value( const String& val ) { m_value = val; m_bValueGiven = true; return *this; }
   CookieParams& path( const String& val ) { m_path = val; return *this; }
   CookieParams& domain( const String& val ) { m_domain = val; return *this; }
   CookieParams& expire_string( const String& val ) { m_expire_string = val; return *this; }

   CookieParams& version( int v ) { m_version = v; return *this; }
   CookieParams& max_age( int v ) { m_max_age = v; return *this; }

   CookieParams& secure( bool v ) { m_secure = v; return *this; }
   CookieParams& httpOnly( bool v ) { m_httpOnly = v; return *this; }

   CookieParams& expire( TimeStamp* v ) { m_expire = v; return *this; }


   String m_value;
   String m_path;
   String m_domain;

   TimeStamp* m_expire;
   String m_expire_string;
   int m_max_age;

   int m_version;

   bool m_secure;
   bool m_httpOnly;
   bool m_bValueGiven;
};


/** Class encapsulating an HTTP reply.
 Final users must re-implement:
 - startCommit() to send the first response reply.
 - commitHeader() that will receive every header and.
 - endCommit() when all the headers are sent.
*/

class Reply
{
public:

   Reply( const CoreClass* base );
   virtual ~Reply();

   /** Sets a cookie.
     Use a variable-parameter idiom provider called CookieParams to configure
     the contents of this cookie.
     \return true if the cookie can be set, false if the headers are sent
    * */
   bool setCookie( const String& cname, const CookieParams& params );
   
   /** Clears a cookie.
     Removes this cookie from the remote browser.
   */
   void clearCookie( const String& cname );
   
   /** Sets the required header to the given value. */
   bool setHeader( const String& fname, const String& value );

   /** Removes the required header.
    \return true if the header is found, false if the headers have been sent or if
            the header is not present.
   */
   bool removeHeader( const String& fname );

   /** Gets the value of the required header.
    \return true if the header name is found, false otherwise.
    */
   bool getHeader( const String& fname, String& value );

   //! Helper to set the content-type
   bool setContentType( const String& type );

   //! Helper to set the content-type
   bool setContentType( const String& type, const String& subtype );

   //! Helper to set the content-type and charset elements.
   bool setContentType( const String& type, const String& subtype, const String& encoding );

   //! Helper to redirect a page
   bool setRedirect( const String& url, uint32 timeout=0 );

   //! Gets all the headers as a dictionary.
   CoreDict* getHeaders();

   //! Send to the final stream (to be re-implemented by the final renderer).
   virtual bool commit();

   //! true if we have perfomed commit
   bool isCommited() const { return m_bHeadersSent; }

   /** Output stream used by this reply.

      It is created by the commit process invoking makeOutputStream().
      In case a specific transcoding is required (i.e. the content-type +
      encoding is set), the stream is wrapped with the correct transcoder
      after makeOutputStream() is invoked.

      The stream is property of this reply and is destroyed with it.
      The VM only knows ReplyStream instances, which are not owning
      this reply nor the final stream. However, a close on one of
      VMachine::stdErr() or VMachine::stdOut() will be reflected here.
   */
   Stream* output() const { return m_output; }


   // Overrides
   virtual CoreObject *clone() const;
   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &prop, Item &value ) const;
   virtual void gcMark( uint32 mark );

protected:

   //! Invoked when the commit operation is about to begin.
   virtual void startCommit() = 0;

   //! Invoked right before startCommit to create an output stream.
   virtual Stream* makeOutputStream() = 0;

   //! Invoked to finalize an header after makeOutputStream.
   virtual void commitHeader( const String& hname, const String& hvalue ) = 0;

   //! Invoked when all the headers are generated.
   virtual void endCommit() = 0;

   // dictionaries
   Utils::StringMap m_mHeaders;
   Utils::StringMap m_mCookies;

   int32  m_nStatus;
   String m_sReason;
   String m_sEncoding;

   bool m_bHeadersSent;
   bool m_bDefaultContent;

   Stream* m_output;
};


}
}

#endif

/* end of reply.h */
