/**
 *  \file gtk_HScrollbar.cpp
 */

#include "gtk_HScrollbar.hpp"

#include "gtk_Adjustment.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void HScrollbar::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_HScrollbar = mod->addClass( "GtkHScrollbar", &HScrollbar::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkScrollbar" ) );
    c_HScrollbar->getClassDef()->addInheritance( in );

    //c_HScrollbar->setWKS( true );
    c_HScrollbar->getClassDef()->factory( &HScrollbar::factory );

}


HScrollbar::HScrollbar( const Falcon::CoreClass* gen, const GtkHScrollbar* bar )
    :
    Gtk::CoreGObject( gen, (GObject*) bar )
{}


Falcon::CoreObject* HScrollbar::factory( const Falcon::CoreClass* gen, void* bar, bool )
{
    return new HScrollbar( gen, (GtkHScrollbar*) bar );
}


/*#
    @class GtkHScrollbar
    @brief A horizontal scrollbar
    @param adjustment the GtkAdjustment to use, or NULL to create a new adjustment

    The GtkHScrollbar widget is a widget arranged horizontally creating a
    scrollbar. See GtkScrollbar for details on scrollbars. GtkAdjustment
    pointers may be added to handle the adjustment of the scrollbar or it
    may be left NULL in which case one will be created for you.
    See GtkScrollbar for a description of what the fields in an adjustment
    represent for a scrollbar.
 */
FALCON_FUNC HScrollbar::init( VMARG )
{
    Item* i_adj = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || !( i_adj->isNil() || ( i_adj->isObject()
        && IS_DERIVED( i_adj, GtkAdjustment ) ) ) )
        throw_inv_params( "[GtkAdjustment]" );
#endif
    GtkAdjustment* adj = i_adj->isNil() ? NULL : GET_ADJUSTMENT( *i_adj );
    MYSELF;
    self->setObject( (GObject*) gtk_hscrollbar_new( adj ) );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
