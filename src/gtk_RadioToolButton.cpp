/**
 *  \file gtk_RadioToolButton.cpp
 */

#include "gtk_RadioToolButton.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void RadioToolButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_RadioToolButton = mod->addClass( "GtkRadioToolButton", &RadioToolButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkToggleToolButton" ) );
    c_RadioToolButton->getClassDef()->addInheritance( in );

    c_RadioToolButton->setWKS( true );
    c_RadioToolButton->getClassDef()->factory( &RadioToolButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "new_from_stock",     &RadioToolButton::new_from_stock },
    //{ "get_group",          &RadioToolButton::get_group },
    //{ "set_group",          &RadioToolButton::set_group },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_RadioToolButton, meth->name, meth->cb );
}


RadioToolButton::RadioToolButton( const Falcon::CoreClass* gen, const GtkRadioToolButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* RadioToolButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new RadioToolButton( gen, (GtkRadioToolButton*) btn );
}


/*#
    @class GtkRadioToolButton
    @brief A toolbar item that contains a radio button
    @param group An existing GtkRadioToolButton, or nil to create a new group of buttons.

    A GtkRadioToolButton is a GtkToolItem that contains a radio button, that is,
    a button that is part of a group of toggle buttons where only one button can
    be active at a time.
 */
FALCON_FUNC RadioToolButton::init( VMARG )
{
    Item* i_grp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grp || !( i_grp->isNil() || ( i_grp->isObject()
        && IS_DERIVED( i_grp, GtkRadioToolButton ) ) ) )
        throw_inv_params( "[GtkRadioToolButton]" );
#endif
    GtkRadioToolButton* grp = i_grp->isNil() ? NULL
                        : (GtkRadioToolButton*) COREGOBJECT( i_grp )->getGObject();
    GtkToolItem* btn = grp ? gtk_radio_tool_button_new_from_widget( grp )
                    : gtk_radio_tool_button_new( NULL );
    MYSELF;
    self->setGObject( (GObject*) btn );
}


/*#
    @method new_from_stock
    @brief Creates a new GtkRadioToolButton, adding it to group.
    @param group An existing GtkRadioToolButton, or nil to create a new group of buttons.
    @param stock_id the name of a stock item
    @return The new GtkRadioToolItem

    The new GtkRadioToolButton will contain an icon and label from the stock
    item indicated by stock_id.
 */
FALCON_FUNC RadioToolButton::new_from_stock( VMARG )
{
    const char* spec = "[GtkRadioToolItem],S";
    Gtk::ArgCheck1 args( vm, spec );
    CoreGObject* o_grp = args.getCoreGObject( 0, false );
    const gchar* stock = args.getCString( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( o_grp && !CoreObject_IS_DERIVED( o_grp, GtkRadioToolButton ) )
        throw_inv_params( spec );
#endif
    GtkRadioToolButton* grp = o_grp ? (GtkRadioToolButton*) o_grp->getGObject() : NULL;
    GtkToolItem* btn = grp ? gtk_radio_tool_button_new_with_stock_from_widget( grp, stock )
                    : gtk_radio_tool_button_new_from_stock( NULL, stock );
    vm->retval( new Gtk::RadioToolButton(
        vm->findWKI( "GtkRadioToolButton" )->asClass(), (GtkRadioToolButton*) btn ) );
}


//FALCON_FUNC RadioToolButton::get_group( VMARG );

//FALCON_FUNC RadioToolButton::set_group( VMARG );


} // Gtk
} // Falcon
