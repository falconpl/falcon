/**
 *  \file gtk_LinkButton.cpp
 */

#include "gtk_LinkButton.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void LinkButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_LinkButton = mod->addClass( "GtkLinkButton", &LinkButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkButton" ) );
    c_LinkButton->getClassDef()->addInheritance( in );

    c_LinkButton->setWKS( true );
    c_LinkButton->getClassDef()->factory( &LinkButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "new_with_label", &LinkButton::new_with_label },
    { "get_uri",        &LinkButton::get_uri },
    { "set_uri",        &LinkButton::set_uri },
    { "set_uri_hook",   &LinkButton::set_uri_hook },
#if GTK_MINOR_VERSION >= 14
    { "get_visited",    &LinkButton::get_visited },
    { "set_visited",    &LinkButton::set_visited },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_LinkButton, meth->name, meth->cb );
}


LinkButton::LinkButton( const Falcon::CoreClass* gen, const GtkLinkButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* LinkButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new LinkButton( gen, (GtkLinkButton*) btn );
}


/*#
    @class GtkLinkButton
    @brief Create buttons bound to a URL
    @param uri a valid URI string

    A GtkLinkButton is a GtkButton with a hyperlink, similar to the one used by
    web browsers, which triggers an action when clicked. It is useful to show
    quick links to resources.

    A link button is created by calling either gtk_link_button_new() or
    gtk_link_button_new_with_label(). If using the former, the URI you pass to
    the constructor is used as a label for the widget.

    The URI bound to a GtkLinkButton can be set specifically using
    gtk_link_button_set_uri(), and retrieved using gtk_link_button_get_uri().

    GtkLinkButton offers a global hook, which is called when the used clicks on
    it: see gtk_link_button_set_uri_hook().
 */
FALCON_FUNC LinkButton::init( VMARG )
{
    Item* i_uri = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_uri || !i_uri->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString uri( i_uri->asString() );
    MYSELF;
    self->setGObject( (GObject*) gtk_link_button_new( uri.c_str() ) );
}


/*#
    @method new_with_label GtkLinkButton
    @brief Creates a new GtkLinkButton containing a label.
    @param uri a valid URI
    @param label the text of the button (or nil).
    @return a new link button widget
 */
FALCON_FUNC LinkButton::new_with_label( VMARG )
{
    Gtk::ArgCheck2 args( vm, "S,[S]" );
    const gchar* uri = args.getCString( 0 );
    const gchar* lbl = args.getCString( 1, false );
    GtkWidget* btn = gtk_link_button_new_with_label( uri, lbl );
    vm->retval( new Gtk::LinkButton( vm->findWKI( "GtkLinkButton" )->asClass(),
                                     (GtkLinkButton*) btn ) );
}


/*#
    @method get_uri GtkLinkButton
    @brief Retrieves the URI set using gtk_link_button_set_uri().
    @return a valid URI
 */
FALCON_FUNC LinkButton::get_uri( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* uri = gtk_link_button_get_uri( (GtkLinkButton*)_obj );
    vm->retval( UTF8String( uri ) );
}


/*#
    @method set_uri GtkLinkButton
    @brief Sets uri as the URI where the GtkLinkButton points.
    @param uri a valid URI.

    As a side-effect this unsets the 'visited' state of the button.
 */
FALCON_FUNC LinkButton::set_uri( VMARG )
{
    Item* i_uri = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_uri || !i_uri->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString uri( i_uri->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_link_button_set_uri( (GtkLinkButton*)_obj, uri.c_str() );
}


/*#
    @method set_uri_hook GtkLinkButton
    @brief Sets func as the function that should be invoked every time a user clicks a GtkLinkButton.
    @param func a function called each time a GtkLinkButton is clicked, or NULL.
    @param user data to be passed to func, or NULL.

    This function is called before every callback registered for the "clicked" signal.

    The function will get the button object as first parameter, the activated link
    as second parameter (string), and user data as third parameter.

    If no uri hook has been set, GTK+ defaults to calling gtk_show_uri().
 */
FALCON_FUNC LinkButton::set_uri_hook( VMARG )
{
    Item* i_func = vm->param( 0 );
    Item* i_data = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_func || !( i_func->isNil() || i_func->isCallable() )
        || !i_data )
        throw_inv_params( "[C,X]" );
#endif
    // release anything previously set
    if ( link_button_uri_hook_func_item )
    {
        gtk_link_button_set_uri_hook( NULL, NULL, NULL );
        delete link_button_uri_hook_func_item;
        link_button_uri_hook_func_item = NULL;
        delete link_button_uri_hook_data_item;
        link_button_uri_hook_data_item = NULL;
    }
    // set new func, if any
    if ( !i_func->isNil() )
    {
        link_button_uri_hook_func_item = new Falcon::GarbageLock( *i_func );
        link_button_uri_hook_data_item = new Falcon::GarbageLock( *i_data );
        gtk_link_button_set_uri_hook( &link_button_uri_hook_func, vm, NULL );
    }
}

Falcon::GarbageLock*    link_button_uri_hook_func_item = NULL;
Falcon::GarbageLock*    link_button_uri_hook_data_item = NULL;

void link_button_uri_hook_func( GtkLinkButton* btn, const gchar* link, gpointer _vm )
{
    assert( link_button_uri_hook_func_item && link_button_uri_hook_data_item );

    VMachine* vm = (VMachine*) _vm;

    vm->pushParam( new Gtk::LinkButton( vm->findWKI( "GtkLinkButton")->asClass(), btn ) );
    vm->pushParam( UTF8String( link ) );
    vm->pushParam( link_button_uri_hook_data_item->item() );
    vm->callItem( link_button_uri_hook_func_item->item(), 3 );
}


#if GTK_MINOR_VERSION >= 14
/*#
    @method get_visited GtkLinkButton
    @brief Retrieves the 'visited' state of the URI where the GtkLinkButton points.
    @return TRUE if the link has been visited, FALSE otherwise

    The button becomes visited when it is clicked. If the URI is changed on the button,
    the 'visited' state is unset again.

    The state may also be changed using gtk_link_button_set_visited().
 */
FALCON_FUNC LinkButton::get_visited( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_link_button_get_visited( (GtkLinkButton*)_obj ) );
}


/*#
    @method set_visited GtkLinkButton
    @brief Sets the 'visited' state of the URI where the GtkLinkButton points.
    @param visited the new 'visited' state
 */
FALCON_FUNC LinkButton::set_visited( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_link_button_set_visited( (GtkLinkButton*)_obj,
                                 (gboolean) i_bool->asBoolean() );
}
#endif // GTK_MINOR_VERSION >= 14


} // Gtk
} // Falcon
