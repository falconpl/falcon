/**
 *  \file gtk_Button.cpp
 */

#include "gtk_Button.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

void Button::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Button = mod->addClass( "Button", &Button::init )
        ->addParam( "label" )
        ->addParam( "mode" );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Container" ) );
    c_Button->getClassDef()->addInheritance( in );

    mod->addClassProperty( c_Button, "NO_MNEMONIC" ).setInteger( 0 ).setReadOnly( true );
    mod->addClassProperty( c_Button, "MNEMONIC" ).setInteger( 1 ).setReadOnly( true );
    mod->addClassProperty( c_Button, "STOCK" ).setInteger( 2 ).setReadOnly( true );

    mod->addClassMethod( c_Button, "signal_clicked",    &Button::signal_clicked );
    //mod->addClassMethod( c_Button, "run",     &Button::run ).asSymbol()
       //->addParam( "window" );


}

/*#
    @class gtk.Button
    @brief A push button
    @optparam label A label string, or a gtk stock id (string)
    @optparam mode (integer) gtk.Button.NO_MNEMONIC (default), or gtk.Button.MNEMONIC, or gtk.Button.STOCK
    @raise ParamError Invalid argument

    If no arguments are given, creates an empty button.
 */

/*#
    @init gtk.Button
 */
FALCON_FUNC Button::init( VMARG )
{
    MYSELF;

    if ( self->getUserData() )
        return;

    Item* i_lbl = vm->param( 0 );

    if ( i_lbl && ( i_lbl->isNil() || !i_lbl->isString() ) )
    {
        throw_inv_params( "[S[,I]]" );
    }

    if ( !i_lbl )
    {
        GtkWidget* btn = gtk_button_new();
        Gtk::internal_add_slot( (GObject*) btn );
        self->setUserData( new GData( (GObject*) btn ) );
        return;
    }

    AutoCString lbl( i_lbl->asString() );
    int mode = 0;

    Item* i_mode = vm->param( 1 );
    if ( i_mode && ( i_mode->isNil() || !i_mode->isInteger() ) )
    {
        throw_inv_params( "[S[,I]]" );
    }
    else if ( i_mode )
    {
        mode = i_mode->asInteger();
        if ( mode < 0 || mode > 2 )
        {
            throw_inv_params( "[S[,I]]" );
        }
    }

    GtkWidget* btn;
    switch ( mode )
    {
    case 0:
        btn = gtk_button_new_with_label( lbl.c_str() );
        break;
    case 1:
        btn = gtk_button_new_with_mnemonic( lbl.c_str() );
        break;
    case 2:
        btn = gtk_button_new_from_stock( lbl.c_str() );
        break;
    default:
        return; // not reached
    }

    Gtk::internal_add_slot( (GObject*) btn );
    self->setUserData( new GData( (GObject*) btn ) );
}


/*#
    @method signal_clicked gtk.Button
    @brief @brief Connect a VMSlot to the button clicked signal and return it
 */
FALCON_FUNC Button::signal_clicked( VMARG )
{
    MYSELF;
    GET_OBJ( self );
    GET_SIGNALS( _obj );
    CoreSlot* ev = get_signal( _obj, _signals,
        "clicked", (void*) &Button::on_clicked, vm );

    Item* it = vm->findWKI( "VMSlot" );
    vm->retval( it->asClass()->createInstance( ev ) );
}


void Button::on_clicked( GtkButton* btn, gpointer _vm )
{
    GET_SIGNALS( btn );
    CoreSlot* cs = _signals->getChild( "clicked", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item* it;

    do
    {
        it = &iter.getCurrent();
        if ( !it->isCallable()
            && ( it->isComposed()
                && !it->asObject()->getMethod( "on_clicked", *it ) ) )
        {
            vm->stdOut()->writeString(
            "[Button::on_clicked] invalid callback (expected callable)\n" );
            return;
        }
        vm->callItem( *it, 0 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


} // Gtk
} // Falcon
