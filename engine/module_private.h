/*
   FALCON - The Falcon Programming Language.
   FILE: module.cpp

   Falcon code unit
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Feb 2011 14:37:57 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_MODULE_PRIVATE_H
#define FALCON_MODULE_PRIVATE_H

#include <falcon/module.h>
#include <falcon/modrequest.h>

#include <falcon/engine.h>
#include <falcon/symbol.h>

#include <deque>
#include <map>
#include <vector>
#include <set>

namespace Falcon {

class Module::Private
{
public:
   typedef std::deque<ModRequest*> ModReqList;
   typedef std::set<Attribute*> AttribSet;

   /** List of requirements.
    
    This list is kept ordered as load order do matters when it comes to
    resolve import order and order of the execution of main code of modules.
    */
   ModReqList m_mrlist;
   
   typedef std::map<String, ModRequest*> ModReqMap;

   /** External modules static requirements, ordered by name.
    
    Name ordering is kept to simplify search for already defined module
    requirements.
    */
   ModReqMap m_mrmap;

   /** Type for external definitions in the Externals map.
    *
    * The first entry is the line where an external entry is declared;
    * the second is the import definition that caused the symbol to be put
    * in the external list. It will be 0 for implicitly imported symbols.
    */

   class ExtDef {
   public:
      int32 m_line;
      ImportDef* m_def;
      const Symbol* m_srcSym;

      ExtDef() {}

      ExtDef(int32 line, ImportDef* idef = 0, const Symbol* srcSym = 0 ):
         m_line(line),
         m_def(idef),
         m_srcSym(srcSym)
      {
         if( srcSym != 0 )
         {
            srcSym->incref();
         }
      }

      ExtDef(int32 line, ImportDef* idef, const String& symName ):
         m_line(line),
         m_def(idef),
         m_srcSym(Engine::getSymbol(symName))
      {
      }

      ExtDef(const ExtDef& other ):
         m_line(other.m_line),
         m_def(other.m_def),
         m_srcSym(other.m_srcSym)
      {
         if ( m_srcSym != 0 ) m_srcSym->incref();
      }

      ~ExtDef() {
         if( m_srcSym != 0 ) m_srcSym->decref();
      }
   };

   typedef std::map<const Symbol*, ExtDef> Externals;
   /* Explicit external requirements.

      List of symbols that have been explicitly imported but not found in the
      module, and that should be resolved at link time.

      To resolve them, the link-time resolver shall use the NSTransMap, which gives
      information on how to retrieve a given symbol.
   */
   Externals m_externals;

   /* Namespace mapping requirements.

      Binds a prefix for namespaced symbols with an import definition describing
      how to find an unknown symbol.
      Since a namespace could include multiple namespaces, as in the following example:
      @code
      import from mod1 in ns
      import ns2.* from mod3 in ns
      import a,b from mod4 in ns
      @endocde

      multiple resolution entires are supported. To search for an unresolved symbol.

      The string "" is used to group the import definitions that have the general
      namespace as target, as the following ones:

      @code
      import a,b,c
      import from mod1
      import ns1.test as test
      @endcode
   */
   typedef std::multimap<String, ImportDef*> NSTransMap;
   NSTransMap m_nsTransMap;

   typedef std::deque<ImportDef*> ImportDefList;
   
   /** List of import definitions.
    
    This is the list of "import" requests. It represents all the "import" directives,
    ordered by their appearance order in the module.
    
    In this list, both "direct" and source import requests are stored.
    
    Direct imports are imports directly generated from code and not from 
    a source script "import" or load directive.
    */
   ImportDefList m_importDefs; 
   
      
   //============================================================
   // Visible entities
   //

   typedef std::map<String, Mantra*> MantraMap;
   MantraMap m_mantras;
   
   // used during deserialization
   uint32 m_symCount;
   std::vector<int32> m_tempExport;
   
   typedef std::deque<Class*> InitList;
   InitList m_initList;

   typedef std::set<String> StringSet;
   StringSet m_istrings;

   Private()
   {}

   ~Private();
};

}

#endif	/* MODULE_PRIVATE_H */

/* end of module_priavate.h */
