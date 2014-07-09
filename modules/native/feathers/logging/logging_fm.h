/*
   FALCON - The Falcon Programming Language.
   FILE: logging_fm.cpp

   Falcon VM interface to logging module -- header.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_FEATHERS_LOGGING_EXT_H
#define FALCON_FEATHERS_LOGGING_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error_base.h>

#define FALCON_FEATHER_LOGGING_NAME "logging"

#define LOGLEVEL_C   0
#define LOGLEVEL_E   1
#define LOGLEVEL_W   2
#define LOGLEVEL_I   3
#define LOGLEVEL_D   4
#define LOGLEVEL_D0  5
#define LOGLEVEL_D1  6
#define LOGLEVEL_D2  7
#define LOGLEVEL_ALL  100

#ifndef FALCON_LOGGING_ERROR_BASE
   #define FALCON_LOGGING_ERROR_BASE         1200
#endif

#define FALCON_LOGGING_ERROR_OPEN  (FALCON_LOGGING_ERROR_BASE + 0)
#define FALCON_LOGGING_ERROR_DESC  "Error opening the logging service"

namespace Falcon {
namespace Feathers {

// ==============================================
// Log area
// ==============================================
class ClassLogArea: public Class
{
public:
   ClassLogArea();
   virtual ~ClassLogArea();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

protected:
   ClassLogArea( const String& name );

private:
   void init();

};


class ClassGeneralLogAreaObj: public ClassLogArea
{
public:
   ClassGeneralLogAreaObj( Class* parent );
   virtual ~ClassGeneralLogAreaObj();

   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
};


// ==============================================
// Log channels
// ==============================================

class ClassLogChannel: public Class
{
public:
   ClassLogChannel();
   virtual ~ClassLogChannel();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

protected:
   ClassLogChannel( const String& name );

private:
   void init();
};


class ClassLogChannelStream: public ClassLogChannel
{
public:
   ClassLogChannelStream( Class* parent );
   virtual ~ClassLogChannelStream();
   virtual void* createInstance() const;
};


class ClassLogChannelEngine: public ClassLogChannel
{
public:
   ClassLogChannelEngine( Class* parent );
   virtual ~ClassLogChannelEngine();
   virtual void* createInstance() const;

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
};


class ClassLogChannelFiles: public ClassLogChannel
{
public:
   ClassLogChannelFiles( Class* parent );
   virtual ~ClassLogChannelFiles();
   virtual void* createInstance() const;
};


class ClassLogChannelSyslog: public ClassLogChannel
{
public:
   ClassLogChannelSyslog( Class* parent );
   virtual ~ClassLogChannelSyslog();
   virtual void* createInstance() const;
};


class LogArea;

class ModuleLogging: public Module
{
public:
   ModuleLogging();
   virtual ~ModuleLogging();

   Class* classLogArea()  const { return m_logArea; }
   Class* classLogChannel()  const { return m_logChannel; }
   LogArea* genericArea() const { return m_generalArea; }

private:

   Class* m_logArea;
   Class* m_logChannel;
   LogArea* m_generalArea;
};


}
}

#endif

/* end of logging_fm.h */
