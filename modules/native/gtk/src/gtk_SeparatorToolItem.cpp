/**
 *  \file gtk_SeparatorToolItem.cpp
 */

#include "gtk_SeparatorToolItem.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void SeparatorToolItem::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_SeparatorToolItem = mod->addClass( "GtkSeparatorToolItem", &SeparatorToolItem::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkToolItem" ) );
    c_SeparatorToolItem->getClassDef()->addInheritance( in );

    //c_SeparatorToolItem->setWKS( true );
    c_SeparatorToolItem->getClassDef()->factory( &SeparatorToolItem::factory );

    mod->addClassMethod( c_SeparatorToolItem, "set_draw", &SeparatorToolItem::set_draw );
    mod->addClassMethod( c_SeparatorToolItem, "get_draw", &SeparatorToolItem::get_draw );
}


SeparatorToolItem::SeparatorToolItem( const Falcon::CoreClass* gen, const GtkSeparatorToolItem* itm )
    :
    Gtk::CoreGObject( gen, (GObject*) itm )
{}


Falcon::CoreObject* SeparatorToolItem::factory( const Falcon::CoreClass* gen, void* itm, bool )
{
    return new SeparatorToolItem( gen, (GtkSeparatorToolItem*) itm );
}


/*#
    @class GtkSeparatorToolItem
    @brief A toolbar item that separates groups of other toolbar items

    A GtkSeparatorItem is a GtkToolItem that separates groups of other GtkToolItems.
    Depending on the theme, a GtkSeparatorToolItem will often look like a vertical
    line on horizontally docked toolbars.

    If the property "expand" is TRUE and the property "draw" is FALSE, a
    GtkSeparatorToolItem will act as a "spring" that forces other items to the
    ends of the toolbar.
 */
FALCON_FUNC SeparatorToolItem::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_separator_tool_item_new() );
}


/*#
    @method set_draw GtkSeparatorToolItem
    @brief Whether item is drawn as a vertical line, or just blank.
    @param draw whether item is drawn as a vertical line

    Setting this to FALSE along with gtk_tool_item_set_expand() is useful to
    create an item that forces following items to the end of the toolbar.
 */
FALCON_FUNC SeparatorToolItem::set_draw( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_separator_tool_item_set_draw( (GtkSeparatorToolItem*)_obj,
                                      i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_draw GtkSeparatorToolItem
    @brief Returns whether item is drawn as a line, or just blank.
    @return TRUE if item is drawn as a line, or just blank.
 */
FALCON_FUNC SeparatorToolItem::get_draw( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_separator_tool_item_get_draw( (GtkSeparatorToolItem*)_obj ) );
}


} // Gtk
} // Falcon
