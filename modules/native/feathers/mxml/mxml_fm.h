/*
   FALCON - The Falcon Programming Language.
   FILE: mxml_FM.h

   Minimal XML module main file - extension definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Mar 2008 18:30:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Minimal XML module main file - extension definitions.
*/

#ifndef FALCON_FEATHERS_MXML_FM_H
#define FALCON_FEATHERS_MXML_FM_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error.h>
#include <falcon/error_base.h>

#ifndef FALCON_MXML_ERROR_BASE
   #define FALCON_MXML_ERROR_BASE            1120
#endif

namespace Falcon {
namespace Feathers {


FALCON_DECLARE_ERROR( MXMLError );


class ClassNode: public Class
{
public:
   ClassNode();
   virtual ~ClassNode();

   virtual int64 occupiedMemory( void* instance ) const;
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};


class ClassDocument: public Class
{
public:
   ClassDocument();
   virtual ~ClassDocument();

   virtual int64 occupiedMemory( void* instance ) const;
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
};


class Enum: public Class
{
public:
   Enum( const String& name );
   virtual ~Enum();
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
};

class ClassStyle: public Enum
{
public:
   ClassStyle();
   virtual ~ClassStyle();
};

class ClassNodeType: public Enum
{
public:
   ClassNodeType();
   virtual ~ClassNodeType();
};

class ClassErrorCode: public Enum
{
public:
   ClassErrorCode();
   virtual ~ClassErrorCode();
};

class ModuleMXML: public Module
{
public:
   ModuleMXML();
   virtual ~ModuleMXML();

   Class* classNode() const { return m_clsNode; }
   Class* classDocument() const { return m_clsDoc; }

private:

   Class* m_clsNode;
   Class* m_clsDoc;
};

}
}

#endif

/* end of mxml_fm.h */
