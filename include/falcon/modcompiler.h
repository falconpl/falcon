/*
   FALCON - The Falcon Programming Language.
   FILE: modcompiler.h

   Module compiler from non-interactive text file.
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

   /** Compile the module read from the given TextReader.
    \param input The TextReader where the input is stored.
    \param uri The physical location of the module
    \param name The logical name of the module.
    \param asFtd true to parse an FTD.
    */
   Module* compile( TextReader* input, const String& uri, const String& name, bool asFtd=false );

   /** Enumerate received errors.
    In case parse returned false, calling this method will provide detailed
    error description for all the errors that have been found.
      \see Enumerator
    */
   inline void enumerateErrors( SourceParser::ErrorEnumerator& e ) const
   {
      m_sp.enumerateErrors( e );
   }
   
   /** Generate an error in case of compilation problems.
    \return an error instance including all the compilation errors or 0 if
    there wasn't any error.
    
    \TODO Return CompileError.
    */
   inline GenericError* makeError() const 
   {
      return m_sp.makeError();
   }

   Module* module() const { return m_module; }
   const SourceParser& sp() const { return m_sp; }
   SourceParser& sp() { return m_sp; }

protected:

   /** Class used to notify the compiler about relevant facts in parsing. */
   class FALCON_DYN_CLASS Context: public ParserContext {
   public:
      Context( ModCompiler* owner );
      virtual ~Context();

      virtual void onInputOver();

      virtual bool onOpenFunc( Function* function );
      virtual void onOpenMethod( Class* cls, Function* function );
      virtual void onCloseFunc( Function* function );
      virtual bool onOpenClass( Class* cls, bool bIsObj );
      virtual void onCloseClass( Class* cls, bool bIsObj );
      virtual bool onAttribute(const String& name, TreeStep* generator, Mantra* target );

      virtual void onNewStatement( TreeStep* stmt );
      virtual void onLoad( const String& path, bool isFsPath );
      virtual bool onImportFrom( ImportDef* def );
      virtual void onExport(const String& symName);
      virtual void onDirective(const String& name, const String& value);
      virtual void onGlobal( const String& name );
      virtual void onGlobalDefined( const String& name, bool& bUnique );
      virtual bool onGlobalAccessed( const String& name );
      virtual Item* getVariableValue( const String& name );
      virtual void onIString(const String& string );

      virtual Item* getValue( const Symbol* name );


   protected:
      ModCompiler* m_owner;
   };

   ModCompiler( ModCompiler::Context* ctx );

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
