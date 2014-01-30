/*
   FALCON - The Falcon Programming Language.
   FILE: importdef.h

   Structure recording the import definition in modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Nov 2011 14:16:51 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_IMPORTDEF_H
#define FALCON_IMPORTDEF_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/sourceref.h>

namespace Falcon {

class Module;
class ModRequest;
class DataWriter;
class DataReader;

/** Structure recording the import definition in modules. 
 This structure holds all the information that can be expressed in an "import"
 or "load" module directive.
*/
class FALCON_DYN_CLASS ImportDef {
public:
   /** Creates an empty import definition. */
   ImportDef();
   
   /** Creates a standard import definition. */
   ImportDef( const String& path, bool isFsPath, const String& symName,
      const String& nsName, bool bIsNS );
   
   ~ImportDef();
   
   /** Sets the target namespace for the imported symbols.
    * @param ns The target namespace.
    *
    * This sets the part after "in" of an import directive:
    * @code
    * import abc,xyz from mod in ns   // "ns" is the target namespace
    * @endcode
    *
    * Setting this will automatically clear the target symbol, as the two
    * declarations are incompatible.
    */
   void setTargetNS( const String& ns );

   /** Sets the target symbol for a single imported source symbol.
    * @param sym The name of the target symbol.
    *
    * This sets the part after "as" of an import directive:
    * @code
    * import abc from mod as myvar   // "myvar" is the target variable
    * @endcode
    *
    * Setting this will automatically clear the target namespace, as the two
    * declarations are incompatible.
    */
   void setTargetSymbol( const String &sym );

   /** Adds a source symbol for the import directive.
    * @param sym The name of the new source symbol.
    *
    * This sets the symbols referred by the import directive.
    * @code
    * import abc, def from mod   // "abc" and "def" are source symbols
    * @endcode
    *
    * Setting this will automatically clear the target namespace, as the two
    * declarations are incompatible.
    */
   ImportDef& addSourceSymbol( const String& sym );

   /** Sets a source module for the import directive.
    * @param src The name of the new source module.
    * @param bIsPath If true, the name expresses a path.
    *
    * @code
    * import from mod            // "mod" is a logical name
    * import from "fm_mod.dll"   // "fm_mod.dll" is a URI name
    * @endcode
    *
    */
   void setImportModule( const String& src, bool bIsPath = false );

   /** Declares this directive as a load directive.
    * @param src The name of the new source module.
    * @param bIsPath If true, the name expresses a path.
    *
    * @code
    * load abc      // "abc" is a logical module
    * load "x.dll"  // "x.dll" is the URI of the module
    * @endcode
    */
   void setLoad( const String& src, bool bIsPath = false );
   
   
   bool isLoad() const { return m_bIsLoad; }
   
   /** Determines whether the source is logical module name or a full URI.
    \return true if the source module is a URI, false if it's a logical name.
    */
   bool isUri() const { return m_bIsUri; }
   
   /** Determines wether the target is a namespace or a single symbol.
    \return true if target() must be intended as a target namespace.
    */
   bool isNameSpace() const { return m_bIsNS; }
   
   /** Count of symbols in the import clause.
    \return Number of symbols declared in the import clause.
    */
   int symbolCount() const;
   
   /** Get the nth symbol imported by this import statement.
    \return The nth symbol name.
    */
   const String& sourceSymbol( int n ) const;
   
   /** Calculates the target symbol.
    * @param i Nth source symbol.
    * @param tgt A target string where to store the calculated name.
    *
    * This removes the source namespace, if any, and sets the target namespace, if any.
    *
    * For instance, in the following declaration:
    * @code
    * import src.a, src.s1.b from mod in myns
    * @endcode
    *
    * the target names will be myns.a and myns.b respectively.
    *
    * */
   void targetSymbol( int i, String& tgt ) const;

   /** Calculates the target symbol.
    *
    * This version of the method returns the string instead
    * of using an already available string as a target parameter.
    *
    * @see targetSymbol( int i, String& tgt )
    */
   String targetSymbol( int i ) const
   {
      String tgt; targetSymbol( i, tgt ); return tgt;
   }
   
   /** Returns the source module for this directive.
    * @return the name of the source module, if given, or an empty string.
    *
    * Use isUri() to determine if the name refers to a logical module name
    * or to a full URI.
    *
    */
   const String& sourceModule() const { return m_module; }

   /** Returns the target symbol or namespace..
    * @return The value of the "in" (target namespace) or "as" (target symbol) clause.
    *
    * Use isNameSpace() to determine if this value represents a target symbol or
    * namespace.
    *
    */
   const String& target() const { return m_tgNameSpace; }
     
   /** Checks if this Include Definition is validly formed.
    *
    * In particular, returns false if the "as/in" clause target:
    * - is filled (not empty)
    * - is a symbol (not a namespace)
    * - there isn't exactly one source symbol.
    *
    * Also, will return false if this is a external import (import a,b,c)
    * and there isn't any symbol.
    * */
   bool isValid() const;
   
   /** Describe this import or load definition as a Falcon directive. */
   void describe( String& tgt ) const;
   
   /** Describe this import or load definition as a Falcon directive. */
   String describe() const {
      String str;
      describe( str );
      return str;
   }
   
   /** Returns true if this is a generic import.
    * \return true if the directive is generic.
    *
    * Generic imports are those directives declared without a specific symbol list,
    * like in the folloing example.
    *
    * @code
    * import from xyz
    * import from xyz in myns
    * @endcode
    *
    * This directives informs the engine that undefined symbols must be looked up
    * in the given source module(s) prior searching them in the global/engine space.
    */
   bool isGeneric() const {return m_symList == 0; }
   
   /** Returns true if the import request is an explicit import request.
    *
    * Explicit import requests are simply request to find a symbol "somewhere",
    * and eventually assign it to a local namespace. They differ from implicit
    * requests as an implicit import is resolved at runtime, while the explicit
    * import is resolved at linktime (and errors are reported prior
    *
    */
   bool isExplicit() const;
   

   const SourceRef& sr() const { return m_sr; }
   SourceRef sr() { return m_sr; }
   
   /** Returns the module request to which this import request refers to.
    
    An import request may or may not refer to a foreign module. If it is a
    load or import/from it does, otherwise it does not.
    
    Module requests are owned by the module owning this same import request.
    This is a mere reference so that it's easier to find the module request
    that may be shared by multiple import requests.
    */
   ModRequest* modReq() const { return m_modreq; }
   
   /** Sets the module request on which this import request insists. */
   void modReq( ModRequest* mr ) { m_modreq = mr; }
   
   /** Position of this entity in the module data.
   Used for simpler serialization, so that it is possible to reference this
   entity in the serialized file.
   */
   int id() const { return m_id; }
   
   /** Sets the position of this entity in the module data.
   Used for simpler serialization, so that it is possible to reference this
   entity in the serialized file.
   */
   void id( int n ) { m_id = n; }
   
   /** Store this definition on a stream. */
   void store(DataWriter* wr) const;
   
   /** Restore this definition from a stream */
   void restore( DataReader* rd);
   
   /** Set to true after the module loader processes this entry. */
   void processed( bool mode ) { m_isProcessed = mode; }
   /** Set to true after the module loader processes this entry. */
   bool processed() const { return m_isProcessed; }

   /** Set to true after the module loader actually loads the required module. */
   void loaded( bool mode ) { m_isLoaded = mode; }
   /** Set to true after the module loader actually loads the required module. */
   bool loaded() const { return m_isLoaded; }

   bool setImportFrom( const String& path, bool isFsPath, const String& symName, const String& nsName, bool bIsNS );

private:
   class SymbolList;
   SymbolList* m_symList;
   
   bool m_bIsLoad;
   bool m_bIsUri;
   bool m_bIsNS;
   bool m_isProcessed;
   bool m_isLoaded;
   
   String m_module;
   String m_tgNameSpace;   
   
   ModRequest* m_modreq; 
   int m_id;
   
   SourceRef m_sr;
};

};

#endif	/* FALCON_IMPORTDEF_H */

/* end of importdef.h */
