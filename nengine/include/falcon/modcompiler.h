/*
   FALCON - The Falcon Programming Language.
   FILE: modcompiler.h

   Interactive compiler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 07 May 2011 16:04:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_MODCOMPILER_H
#define	FALCON_MODCOMPILER_H

#include <falcon/setup.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/syntree.h>

namespace Falcon {

class Module;
class VMachine;
class StringStream;
class TextReader;

/** Class encapsulating a static module compiler.

 A module compiler is a compiler generating a stand-alone module out of
 a source file.

 A Falcon module can then be serialized to a stream for later usage, or
 can be linked immediately in a virtual machine. Once linked, the module
 cannot be used elsewhere. However, a module can be cloned before linking
 to be used multiple times in a single program.

 A module compiler can be used multiple times.
 */
class FALCON_DYN_CLASS ModCompiler
{

public:
   ModCompiler();
   virtual ~ModCompiler();

   Module* compile( TextReader* input );

private:

   /** Class used to notify the compiler about relevant facts in parsing. */
   class Context: public ParserContext {
   public:
      Context( ModCompiler* owner );
      virtual ~Context();

      virtual void onInputOver();
      virtual void onNewFunc( Function* function, GlobalSymbol* gs=0 );
      virtual void onNewClass( Class* cls, bool bIsObj, GlobalSymbol* gs=0 );
      virtual void onNewStatement( Statement* stmt );
      virtual void onLoad( const String& path, bool isFsPath );
      virtual void onImportFrom( const String& path, bool isFsPath, const String& symName,
            const String& asName, const String &inName );
      virtual void onImport(const String& symName );
      virtual void onExport(const String& symName);
      virtual void onDirective(const String& name, const String& value);
      virtual void onGlobal( const String& name );
      virtual Symbol* onUndefinedSymbol( const String& name );
      virtual GlobalSymbol* onGlobalDefined( const String& name, bool& bUnique );
      virtual bool onUnknownSymbol( UnknownSymbol* sym );
      virtual void onStaticData( Class* cls, void* data );
      virtual void onInheritance( Inheritance* inh  );

   private:
      ModCompiler* m_owner;
   };

   SourceParser m_sp;

   /** Used to keep non-transient data. */
   Module* m_module;

   // better for the context to be a pointer, so we can control it's init order.
   Context* m_ctx;
   friend class Context;
};

}

#endif	/* FALCON_MODCOMPILER_H */

/* end of modcompiler.h */
