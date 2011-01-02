/*
 *  Falcon DataMatrix - Module definition
 */

#include "dmtx_ext.h"
#include "dmtx_mod.h"
#include "dmtx_srv.h"
//#include "dmtx_st.h"
#include "version.h"

#include <falcon/module.h>

/*#
   @module dmtx DataMatrix module
   @brief DataMatrix utilities.
*/

/*
    TODO LIST...
 */

using namespace Falcon;

Falcon::DataMatrixService theDataMatrixService;


FALCON_MODULE_DECL
{
    Falcon::Module *self = new Falcon::Module();
    self->name( "dmtx" );
    self->engineVersion( FALCON_VERSION_NUM );
    self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

    //#include "dmtx_st.h"

    // *** Constants ***
    self->addConstant( "DmtxUndefined",         (int64) -1 );
    // Schemes...
    self->addConstant( "DmtxSchemeAutoFast",    (int64) -2 );
    self->addConstant( "DmtxSchemeAutoBest",    (int64) -1 );
    self->addConstant( "DmtxSchemeAscii",       (int64) 0 );
    self->addConstant( "DmtxSchemeC40",         (int64) 1 );
    self->addConstant( "DmtxSchemeText",        (int64) 2 );
    self->addConstant( "DmtxSchemeX12",         (int64) 3 );
    self->addConstant( "DmtxSchemeEdifact",     (int64) 4 );
    self->addConstant( "DmtxSchemeBase256",     (int64) 5 );
    // Shapes...
    self->addConstant( "DmtxSymbolRectAuto",    (int64) -3 );
    self->addConstant( "DmtxSymbolSquareAuto",  (int64) -2 );
    self->addConstant( "DmtxSymbolShapeAuto",   (int64) -1 );
    self->addConstant( "DmtxSymbol10x10",       (int64) 0 );
    self->addConstant( "DmtxSymbol12x12",       (int64) 1 );
    self->addConstant( "DmtxSymbol14x14",       (int64) 2 );
    self->addConstant( "DmtxSymbol16x16",       (int64) 3 );
    self->addConstant( "DmtxSymbol18x18",       (int64) 4 );
    self->addConstant( "DmtxSymbol20x20",       (int64) 5 );
    self->addConstant( "DmtxSymbol22x22",       (int64) 6 );
    self->addConstant( "DmtxSymbol24x24",       (int64) 7 );
    self->addConstant( "DmtxSymbol26x26",       (int64) 8 );
    self->addConstant( "DmtxSymbol32x32",       (int64) 9 );
    self->addConstant( "DmtxSymbol36x36",       (int64) 10 );
    self->addConstant( "DmtxSymbol40x40",       (int64) 11 );
    self->addConstant( "DmtxSymbol44x44",       (int64) 12 );
    self->addConstant( "DmtxSymbol48x48",       (int64) 13 );
    self->addConstant( "DmtxSymbol52x52",       (int64) 14 );
    self->addConstant( "DmtxSymbol64x64",       (int64) 15 );
    self->addConstant( "DmtxSymbol72x72",       (int64) 16 );
    self->addConstant( "DmtxSymbol80x80",       (int64) 17 );
    self->addConstant( "DmtxSymbol88x88",       (int64) 18 );
    self->addConstant( "DmtxSymbol96x96",       (int64) 19 );
    self->addConstant( "DmtxSymbol104x104",     (int64) 20 );
    self->addConstant( "DmtxSymbol120x120",     (int64) 21 );
    self->addConstant( "DmtxSymbol132x132",     (int64) 22 );
    self->addConstant( "DmtxSymbol144x144",     (int64) 23 );
    self->addConstant( "DmtxSymbol8x18",        (int64) 24 );
    self->addConstant( "DmtxSymbol8x32",        (int64) 25 );
    self->addConstant( "DmtxSymbol12x26",       (int64) 26 );
    self->addConstant( "DmtxSymbol12x36",       (int64) 27 );
    self->addConstant( "DmtxSymbol16x36",       (int64) 28 );
    self->addConstant( "DmtxSymbol16x48",       (int64) 29 );

    // DataMatrix class
    Falcon::Symbol* dmtx_cls = self->addClass( "DataMatrix",
                                               Falcon::Ext::DataMatrix_init );
    dmtx_cls->setWKS( true );
    dmtx_cls->getClassDef()->factory( Falcon::Dmtx::DataMatrix::factory );

    self->addClassProperty( dmtx_cls, "module_size" );
    self->addClassProperty( dmtx_cls, "margin_size" );
    self->addClassProperty( dmtx_cls, "gap_size" );
    self->addClassProperty( dmtx_cls, "scheme" );
    self->addClassProperty( dmtx_cls, "shape" );

    self->addClassProperty( dmtx_cls, "timeout" );
    self->addClassProperty( dmtx_cls, "shrink" );
    self->addClassProperty( dmtx_cls, "deviation" );
    self->addClassProperty( dmtx_cls, "threshold" );
    self->addClassProperty( dmtx_cls, "min_edge" );
    self->addClassProperty( dmtx_cls, "max_edge" );
    self->addClassProperty( dmtx_cls, "corrections" );
    self->addClassProperty( dmtx_cls, "max_count" );

    self->addClassMethod( dmtx_cls, "encode", Falcon::Ext::DataMatrix_encode );
    self->addClassMethod( dmtx_cls, "decode", Falcon::Ext::DataMatrix_decode );

    // DataMatrixError class
    Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
    Falcon::Symbol *err_cls = self->addClass( "DataMatrixError", &Falcon::Ext::DataMatrixError_init );
    err_cls->setWKS( true );
    err_cls->getClassDef()->addInheritance( new Falcon::InheritDef( error_class ) );

    // service publication
    self->publishService( &theDataMatrixService );

    return self;
}
