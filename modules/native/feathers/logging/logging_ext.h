/*
   FALCON - The Falcon Programming Language.
   FILE: logging_ext.cpp

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

#define LOGLEVEL_C   0
#define LOGLEVEL_E   1
#define LOGLEVEL_W   2
#define LOGLEVEL_I   3
#define LOGLEVEL_D   4
#define LOGLEVEL_D0  5
#define LOGLEVEL_D1  6
#define LOGLEVEL_D2  7

namespace Falcon {


namespace Ext {

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


class ClassGeneralLogAreaObj: public Class
{
public:
   ClassGeneralLogAreaObj( Class* parent );
   virtual ~ClassGeneralLogAreaObj();

   virtual void* createInstance() const;
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

protected:
   ClassLogChannel( const String& name );
};


class ClassLogChannelTW: public Class
{
public:
   ClassLogChannelTW( Class* parent );
   virtual ~ClassLogChannelTW();

   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;
};


class ClassLogChannelEngine: public Class
{
public:
   ClassLogChannelEngine( Class* parent );
   virtual ~ClassLogChannelEngine();

   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;
};


class ClassLogChannelFiles: public Class
{
public:
   ClassLogChannelFiles( Class* parent );
   virtual ~ClassLogChannelFiles();

   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;
};


class ClassLogChannelSyslog: public Class
{
public:
   ClassLogChannelSyslog( Class* parent );
   virtual ~ClassLogChannelSyslog();

   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;
};

}
}

#endif

/* end of logging_ext.h */
