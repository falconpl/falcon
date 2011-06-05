/**
 *  \file gtk_VScrollbar.cpp
 */

#include "gtk_VScrollbar.hpp"

#include "gtk_Adjustment.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void VScrollbar::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_VScrollbar = mod->addClass( "GtkVScrollbar", &VScrollbar::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkScrollbar" ) );
    c_VScrollbar->getClassDef()->addInheritance( in );

    c_VScrollbar->setWKS( true );
    c_VScrollbar->getClassDef()->factory( &VScrollbar::factory );

}


VScrollbar::VScrollbar( const Falcon::CoreClass* gen, const GtkVScrollbar* bar )
    :
    Gtk::CoreGObject( gen, (GObject*) bar )
{}


Falcon::CoreObject* VScrollbar::factory( const Falcon::CoreClass* gen, void* bar, bool )
{
    return new VScrollbar( gen, (GtkVScrollbar*) bar );
}


/*#
    @class GtkVScrollbar
    @brief A vertical scrollbar
    @param adjustment the GtkAdjustment to use, or NULL to create a new adjustment

    The GtkVScrollbar widget is a widget arranged vertically creating a scrollbar.
    See GtkScrollbar for details on scrollbars. GtkAdjustment pointers may be added
    to handle the adjustment of the scrollbar or it may be left NULL in which case
    one will be created for you. See GtkScrollbar for a description of what the
    fields in an adjustment represent for a scrollbar.
 */
FALCON_FUNC VScrollbar::init( VMARG )
{
    Item* i_adj = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || !( i_adj->isNil() || ( i_adj->isObject()
        && IS_DERIVED( i_adj, GtkAdjustment ) ) ) )
        throw_inv_params( "[GtkAdjustment]" );
#endif
    GtkAdjustment* adj = i_adj->isNil() ? NULL : GET_ADJUSTMENT( *i_adj );
    MYSELF;
    self->setObject( (GObject*) gtk_vscrollbar_new( adj ) );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
