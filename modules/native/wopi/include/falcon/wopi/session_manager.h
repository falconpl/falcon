/*
   FALCON - The Falcon Programming Language.
   FILE: session_manager.h

   Falcon Web Oriented Programming Interface
   Session manager.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Feb 2010 23:17:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _SESSION_MANAGER_H
#define _SESSION_MANAGER_H

#define FALCON_WOPI_SESSION_TO_ATTRIB "wopi_sessionTO"

#include <falcon/engine.h>
#include <map>
#include <list>

namespace Falcon {
namespace WOPI {

/** Abstract class representing a single session data.

    Subclasses should reimplement the methods assign, release and dispose
    to manage system resources related with this session data.
*/
class SessionData
{
public:
   SessionData( const String& SID );
   virtual ~SessionData();

   /** Resumes a suspended  session.

       Suppose a user code restart (i.e. after a server HUP, or the
       new invocation of a CGI script). The user may ask for a session
       item via a HTTP query which is not currently existing in the
       session map but that is available on some persistent storage.

       Resume is invoked after a getSession has not found this session
       object in the session map, and thus has created it.

       It may return false if the resume fails, or if this particular
       subclass of SessionData is not resume-able (i.e. if it's fully
       memory based, not providing any persistency).

       This method must also have the acquire semantic (possibly calling it).
   */
   virtual bool resume() = 0;

   /** Stores a session to persistent media.
   */
   virtual bool store() = 0;

   /** Disposes this session.

       This is call right before the destructor to allow the
       session to try to perform system operations that may fail.

       Subclasses should implement this function so that all the
       system resources allocated to manage this session are removed.
   */
   virtual bool dispose() = 0;

   /** Assigns the session to a certain assignee.

       An assigned session never expires and cannot be assigned to another
       request (getSession in the session manager shall fail).

       Subclasses must remember to update the m_assignee field.

       \note This should be called in the session lock context.
   */
   void assign( uint32 assignee ) { m_assignee = assignee; }

   /** Releases a session.

       \note This should be called in the session lock context.
   */
   void release() { m_assignee = 0; }

   /** Indicates that this session has been used now.

       This marks the last time this session is used, preventing
       early expiration.
    */
   void touch();

   void setError( int e );
   void setError( const String& edesc, int e = 0 );


   //====================================================================
   // accessors
   CoreDict* data() const { return m_dataLock.item().asDict(); }
   bool isAssigned() const { return m_assignee != 0; }
   uint32 assigned() const { return m_assignee; }

   /** Returns the last time (relative to Falcon::Sys::_seconds) this session has been used.
   */
   numeric lastTouched() const { return m_touchTime; }

   const String& errorDesc() const { return m_errorDesc; }
   int lastError() const { return m_lastError; }

   const String& sID() const { return m_sID; }

   bool isInvalid() const { return m_bInvalid; }
   void setInvalid() { m_bInvalid = true; }

   /** The weak reference to this class */
   class WeakRef
   {
   public:
      WeakRef( SessionData* r ):
         m_ref(r),
         m_bDropped(false)
      {
      }

      void onDestroy()
      {
         if( m_bDropped )
            delete this;
         else
            m_ref = 0;
      }

      SessionData* get() const { return m_ref; }
   
      void dropped()
      {
         if ( m_ref == 0 )
            delete this;
         else
            m_bDropped = true;
      }

   private:
      SessionData* m_ref;
      bool m_bDropped;
   };

   /** Gets a weak reference to this class.
      The reference pointer is turned into 0 when the object is destroyed,
      or otherwise invalidated.
   */
   WeakRef* getWeakRef();

   /** Called during invalidation or destruction to clear weak references. 
      Actually, it should be called when the session manager is locked.
   */
   void clearRefs();

protected:
   GarbageLock m_dataLock;
   uint32 m_assignee;

private:
   numeric m_touchTime;

   int m_lastError;
   String m_errorDesc;
   String m_sID;
   bool m_bInvalid;

   mutable std::list<WeakRef*> m_reflist;
};


/** Repository for session variables. */
class SessionManager
{
public:
   SessionManager();
   virtual ~SessionManager();

   /** Gets an existing session for a particular user.
      If the required session doesn't exist, a new session is created.

      On first call, the token shall be zero. An unique session token is
      then given, and can be used to ask for new sessions opened in the
      same context.

      When the users of the session created in this context is terminated,
      the session token can be used to close all the related sessions.

      \note If the session doesn't exist, getSession tries to create it
            via createSession() and then SessionData::resume().

      \param sSID Session identificator.
      \parma token A session owner identifier.
      \return A core dictionary containing the session data.
   */
   SessionData* getSession( const String& sSID, uint32 token );

   /** Starts a new session.

       This method is similar to getSession, but it creates a new
       session, without calling SessionData::resume()
    */
   SessionData* startSession( uint32 token );

   /** Tries to start a new session.
    *
      This version allows an external source to decide the ID of the new
      session. It is useful in case of unique IDs being sent during multiple
      step authentication schemes as OAUTH.

   */
   SessionData* startSession( uint32 token, const Falcon::String &sSID );

   /** Clears all the sessions that have been opened by the given user.

       Call this when the session user associated with this token doesn't
       exist anymore.

       All the released sessions are first touched, then stored and finally
       released.

       \note This function must be called in the same thread of expireOldSessions
       to avoid concurrent expiration & storage (there can't be any mutex protection
       against this. A refcount protection may prevent crashes, but may cause
       dirty files and memory leaks to be left on the ground).
   */
   bool releaseSessions( uint32 token );

   /** Creates an unique session token. */
   uint32 getSessionToken();

   /** Closes a session explicitly, freeing its data. */
   bool closeSession( const String& sSID, uint32 token );

   void timeout( uint32 to ) { m_nSessionTimeout = to; }
   uint32 timeout() const { return m_nSessionTimeout; }

   /** Prepare first execution.
       To be called after a complete configuration.
    */
   virtual void startup() = 0;

   virtual void configFromModule( const Module* mod );

protected:
   /** Creates a session adequate for this session manager.
       Must be fast.
   */
   virtual SessionData* createSession( const String& sSID ) = 0;

   /** Creates a unique ID and returning atomically a garbage lock.

      This also creates an empty session record in the SessionMap.
   */
   SessionData* createUniqueId( String& sSID );

private:

   /**
      Delete sessions that are expired in time.
      
      \note This function must be called in the same thread of releaseSessions()
       to avoid concurrent expiration & storage (there can't be any mutex protection
       against this. A refcount protection may prevent crashes, but may cause
       dirty files and memory leaks to be left on the ground).
   */
 
   void expireOldSessions();

   typedef std::map<String, SessionData*> SessionMap;

   typedef std::list<SessionData::WeakRef*> SessionList;
   typedef std::map<uint32, SessionList> SessionUserMap;
   typedef std::multimap<double, SessionData::WeakRef*> ExpirationMap;

   //Rightful owner of sessions
   SessionMap m_smap;

   // Owning weakrefs
   SessionUserMap m_susers;
   ExpirationMap m_expmap;
   mutable Falcon::Mutex m_mtx;

   uint32 m_nLastToken;
   uint32 m_nSessionTimeout;
};

}
}

#endif

/* end of session_manager.h */
