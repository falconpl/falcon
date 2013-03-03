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
class ImportDef {
public:
   /** Creates an empty import definition. */
   ImportDef();
   
   /** Creates a standard import definition. */
   ImportDef( const String& path, bool isFsPath, const String& symName,
      const String& nsName, bool bIsNS );
   
   ~ImportDef();
   
   void setTargetNS( const String& ns );
   void setTargetSymbol( const String &sym );
   void addSourceSymbol( const String& sym );
   void setImportModule( const String& src, bool bIsPath = false );
   void setLoad( const String& src, bool bIsPath = false );
   
   bool setImportFrom( const String& path, bool isFsPath, const String& symName,
      const String& nsName, bool bIsNS );
   
   bool isLoad() const { return m_bIsLoad; }
   
   /** Determines wether the source is logical module name or a full URI.
    \return true if target() must be intended as a target namespace.
    */
   bool isUri() const { return m_bIsUri; }
   
   /** Determines wether the target is a namespace or a single symbol.
    \return true if target() must be intended as a target namespace.
    */
   bool isNameSpace() const { return m_bIsNS; }
   
   /** Count of symbols in the import clause.
    \return true if target() must be intended as a target namespace.
    */
   int symbolCount() const;
   
   /** Get the nth symbol imported by this import statement.
    \return true if target() must be intended as a target namespace.
    */
   const String& sourceSymbol( int n ) const;
   
   /** Calculates the target symbol. */
   void targetSymbol( int i, String& tgt ) const;
   String targetSymbol( int i ) const
   {
      String tgt; targetSymbol( i, tgt ); return tgt;
   }
   
   const String& sourceModule() const { return m_source; }
   const String& target() const { return m_tgNameSpace; }
     
   /** Checks if this Include Definition is validly formed. */
   bool isValid() const;
   
   /** Describe this import or load definition as a Falcon directive. */
   void describe( String& tgt ) const;
   
   /** Describe this import or load definition as a Falcon directive. */
   String describe() const {
      String str;
      describe( str );
      return str;
   }
   
   /** Returns true if this is a generic (without symbols) import. */
   bool isGeneric() const {return m_symList == 0; }
   
   /** Set this as a direct import request. 
    \param symName the name of the symbol to be searched on the source module.
    \param modName The name of the module where the symbol is to be searched.
    \param bIsURI If true, the modName parameter is intended as a VFS URI,
    otherwise is a logical name.
    
    Direct import request are created explicity from code and third party
    modules when the module is synthesized from binary data. They passed 
    directly to a resolve handler when the symbol is resolved.
    
    They are equivalent to the directive "import sym from mod", but won't
    create a symbol in the host module; instead, when they are resolved
    this usually triggers a set of Requirement callbacks.
    
    This version requires explicitly a symbol to be found in a module.
    */
   void setDirect( const String& symName, const String& modName, bool bIsURI );
   
   /** Set this as a direct import request. 
    \param symName the name of the symbol to be searched on the source module.
    
    Direct import request are created explicity from code and third party
    modules when the module is synthesized from binary data. They passed 
    directly to a resolve handler when the symbol is resolved.
    
    This version searches the symbol on exported symbols of the loaded modules in
    the host ModSpace. Notice that load order declaration is respected; when
    unsure, put a load ImportDef before this direct import reqeust on the 
    target module.
    
    \note Direct imports are not shown when rendering a module to string.
    */
   void setDirect( const String& symName );
   
   /** Return true if this is a direct import request.
    \return True if this is a direct import request.
    \see setDirect
    */
   bool isDirect() const {return m_bIsDirect; }
   
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
   
   void processed( bool mode ) { m_isProcessed = mode; }
   bool processed() const { return m_isProcessed; }
   void loaded( bool mode ) { m_isLoaded = mode; }
   bool loaded() const { return m_isLoaded; }

private:
   class SymbolList;
   SymbolList* m_symList;
   
   bool m_bIsLoad;
   bool m_bIsUri;
   bool m_bIsNS;
   bool m_bIsDirect;
   bool m_isProcessed;
   bool m_isLoaded;
   
   String m_source;
   String m_tgNameSpace;   
   
   ModRequest* m_modreq; 
   int m_id;
   
   SourceRef m_sr;
};

};

#endif	/* FALCON_IMPORTDEF_H */

/* end of importdef.h */
