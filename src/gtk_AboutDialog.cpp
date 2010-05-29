/**
 *  \file gtk_AboutDialog.cpp
 */

#include "gtk_AboutDialog.hpp"

#if GTK_MINOR_VERSION >= 6

#include "gdk_Pixbuf.hpp"


namespace Falcon {
namespace Gtk {


/**
 *  \brief module init
 */
void AboutDialog::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_AboutDialog = mod->addClass( "GtkAboutDialog", &AboutDialog::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkDialog" ) );
    c_AboutDialog->getClassDef()->addInheritance( in );

    c_AboutDialog->setWKS( true );
    c_AboutDialog->getClassDef()->factory( &AboutDialog::factory );

    Gtk::MethodTab methods[] =
    {
    { "get_name",           &AboutDialog::get_name },
    { "set_name",           &AboutDialog::set_name },
#if GTK_MINOR_VERSION >= 12
    { "get_program_name",   &AboutDialog::get_program_name },
    { "set_program_name",   &AboutDialog::set_program_name },
#endif
    { "get_version",        &AboutDialog::get_version },
    { "set_version",        &AboutDialog::set_version },
    { "get_copyright",      &AboutDialog::get_copyright },
    { "set_copyright",      &AboutDialog::set_copyright },
    { "get_comments",       &AboutDialog::get_comments },
    { "set_comments",       &AboutDialog::set_comments },
    { "get_license",        &AboutDialog::get_license },
    { "set_license",        &AboutDialog::set_license },
#if GTK_MINOR_VERSION >= 8
    { "get_wrap_license",   &AboutDialog::get_wrap_license },
    { "set_wrap_license",   &AboutDialog::set_wrap_license },
#endif
    { "get_website",        &AboutDialog::get_website },
    { "set_website",        &AboutDialog::set_website },
    { "get_website_label",  &AboutDialog::get_website_label },
    { "set_website_label",  &AboutDialog::set_website_label },
    { "get_authors",        &AboutDialog::get_authors },
    { "set_authors",        &AboutDialog::set_authors },
    { "get_artists",        &AboutDialog::get_artists },
    { "set_artists",        &AboutDialog::set_artists },
    { "get_documenters",    &AboutDialog::get_documenters },
    { "set_documenters",    &AboutDialog::set_documenters },
    { "get_translator_credits",&AboutDialog::get_translator_credits },
    { "set_translator_credits",&AboutDialog::set_translator_credits },
    { "get_logo",           &AboutDialog::get_logo },
    { "set_logo",           &AboutDialog::set_logo },
    { "get_logo_icon_name", &AboutDialog::get_logo_icon_name },
    { "set_logo_icon_name", &AboutDialog::set_logo_icon_name },
    { "set_email_hook",     &AboutDialog::set_email_hook },
    { "set_url_hook",       &AboutDialog::set_url_hook },
    //{ "show_about_dialog",  &AboutDialog::show_about_dialog },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_AboutDialog, meth->name, meth->cb );
}


AboutDialog::AboutDialog( const Falcon::CoreClass* gen, const GtkAboutDialog* dlg )
    :
    Gtk::CoreGObject( gen, (GObject*) dlg )
{}


Falcon::CoreObject* AboutDialog::factory( const Falcon::CoreClass* gen, void* dlg, bool )
{
    return new AboutDialog( gen, (GtkAboutDialog*) dlg );
}


/*#
    @class GtkAboutDialog
    @brief Display information about an application

    The GtkAboutDialog offers a simple way to display information about a program
    like its logo, name, copyright, website and license. It is also possible to give
    credits to the authors, documenters, translators and artists who have worked on
    the program. An about dialog is typically opened when the user selects the About
    option from the Help menu. All parts of the dialog are optional.

    About dialog often contain links and email addresses. GtkAboutDialog supports
    this by offering global hooks, which are called when the user clicks on a link
    or email address, see set_email_hook() and set_url_hook(). Email addresses in
    the authors, documenters and artists properties are recognized by looking for
    <user@host>, URLs are recognized by looking for http://url, with url extending
    to the next space, tab or line break.

    [...]

 */
FALCON_FUNC AboutDialog::init( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    self->setGObject( (GObject*) gtk_about_dialog_new() );
}


/*#
    @method get_name GtkAboutDialog
    @brief  Returns the program name displayed in the about dialog.
    @return The program name.

    get_name has been deprecated since version 2.12 and should not be used in
    newly-written code. Use get_program_name() instead.
 */
FALCON_FUNC AboutDialog::get_name( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* nm = gtk_about_dialog_get_name( (GtkAboutDialog*)_obj );
    vm->retval( UTF8String( nm ) );
}


/*#
    @method set_name GtkAboutDialog
    @brief Sets the name to display in the about dialog.
    @param name the program name (or nil).

    If this is not set, it defaults to g_get_application_name().

    set_name has been deprecated since version 2.12 and should not be used in
    newly-written code. Use set_program_name() instead.
 */
FALCON_FUNC AboutDialog::set_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const char* nm = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_name( (GtkAboutDialog*)_obj, nm );
}


#if GTK_MINOR_VERSION >= 12
/*#
    @method get_program_name GtkAboutDialog
    @brief Returns the program name displayed in the about dialog.
    @return The program name.
 */
FALCON_FUNC AboutDialog::get_program_name( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* nm = gtk_about_dialog_get_program_name( (GtkAboutDialog*)_obj );
    vm->retval( UTF8String( nm ) );
}


/*#
    @method set_program_name GtkAboutDialog
    @brief Sets the name to display in the about dialog.
    @param name the program name

    If this is not set, it defaults to g_get_application_name().
 */
FALCON_FUNC AboutDialog::set_program_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const char* nm = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_program_name( (GtkAboutDialog*)_obj, nm );
}
#endif // GTK_MINOR_VERSION >= 12


/*#
    @method get_version GtkAboutDialog
    @brief Returns the version string.
    @return The version string.
 */
FALCON_FUNC AboutDialog::get_version( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* ver = gtk_about_dialog_get_version( (GtkAboutDialog*)_obj );
    vm->retval( UTF8String( ver ) );
}


/*#
    @method set_version GtkAboutDialog
    @brief Sets the version string to display in the about dialog.
    @param the version string (or nil).
 */
FALCON_FUNC AboutDialog::set_version( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const char* ver = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_version( (GtkAboutDialog*)_obj, ver );
}


/*#
    @method get_copyright GtkAboutDialog
    @brief Returns the copyright string.
    @return The copyright string.
 */
FALCON_FUNC AboutDialog::get_copyright( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* cr = gtk_about_dialog_get_copyright( (GtkAboutDialog*)_obj );
    vm->retval( UTF8String( cr ) );
}


/*#
    @method set_copyright GtkAboutDialog
    @brief Sets the copyright string to display in the about dialog.
    @param copyright the copyright string (or nil).

    This should be a short string of one or two lines.
 */
FALCON_FUNC AboutDialog::set_copyright( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const char* cr = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_copyright( (GtkAboutDialog*)_obj, cr );
}


/*#
    @method get_comments GtkAboutDialog
    @brief Returns the comments string.
    @return The comments.
 */
FALCON_FUNC AboutDialog::get_comments( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* com = gtk_about_dialog_get_comments( (GtkAboutDialog*)_obj );
    vm->retval( UTF8String( com ) );
}


/*#
    @method set_comments GtkAboutDialog
    @brief Sets the comments string to display in the about dialog.
    @param comments a comments string (or nil).

    This should be a short string of one or two lines.
 */
FALCON_FUNC AboutDialog::set_comments( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const char* com = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_comments( (GtkAboutDialog*)_obj, com );
}


/*#
    @method get_license GtkAboutDialog
    @brief Returns the license information
    @return The license information.
 */
FALCON_FUNC AboutDialog::get_license( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* lic = gtk_about_dialog_get_license( (GtkAboutDialog*)_obj );
    vm->retval( UTF8String( lic ) );
}


/*#
    @method set_license GtkAboutDialog
    @brief Sets the license information to be displayed in the secondary license dialog.
    @param license the license information or nil.

    If license is nil, the license button is hidden.
 */
FALCON_FUNC AboutDialog::set_license( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const char* lic = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_license( (GtkAboutDialog*)_obj, lic );
}


#if GTK_MINOR_VERSION >= 8
/*#
    @method get_wrap_license GtkAboutDialog
    @brief Returns whether the license text in about is automatically wrapped.
    @return true if the license text is wrapped
 */
FALCON_FUNC AboutDialog::get_wrap_license( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_about_dialog_get_wrap_license( (GtkAboutDialog*)_obj ) );
}


/*#
    @method set_wrap_license GtkAboutDialog
    @brief Sets whether the license text in about is automatically wrapped.
    @param wrap_license wether to wrap the license
 */
FALCON_FUNC AboutDialog::set_wrap_license( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_wrap_license( (GtkAboutDialog*)_obj,
                                       (gboolean) i_bool->asBoolean() );
}
#endif // GTK_MINOR_VERSION >= 8


/*#
    @method get_website GtkAboutDialog
    @brief Returns the website URL.
    @return The website URL.
 */
FALCON_FUNC AboutDialog::get_website( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* web = gtk_about_dialog_get_website( (GtkAboutDialog*)_obj );
    vm->retval( UTF8String( web ) );
}


/*#
    @method set_website
    @brief Sets the URL to use for the website link.
    @param a URL string starting with "http://" (or nil).

    Note that that the hook functions need to be set up before calling this function.
 */
FALCON_FUNC AboutDialog::set_website( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const char* web = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_website( (GtkAboutDialog*)_obj, web );
}


/*#
    @method get_website_label GtkAboutDialog
    @brief Returns the label used for the website link.
    @return The label used for the website link.
 */
FALCON_FUNC AboutDialog::get_website_label( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* lbl = gtk_about_dialog_get_website_label( (GtkAboutDialog*)_obj );
    vm->retval( UTF8String( lbl ) );
}


/*#
    @method set_website_label GtkAboutDialog
    @brief Sets the label to be used for the website link. It defaults to the website URL.
    @param website_label the label used for the website link
 */
FALCON_FUNC AboutDialog::set_website_label( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const char* lbl = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_website_label( (GtkAboutDialog*)_obj, lbl );
}


/*#
    @method get_authors GtkAboutDialog
    @brief Returns the string which are displayed in the authors tab of the secondary credits dialog.
    @return A string array containing the authors.
 */
FALCON_FUNC AboutDialog::get_authors( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* const *authors = gtk_about_dialog_get_authors( (GtkAboutDialog*)_obj );
    int sz = 0;
    int i = 0;
    for ( gchar* auth = (gchar*) authors[i++]; auth; auth = (gchar*) authors[i++] )
        ++sz;
    CoreArray* arr = new CoreArray( sz );
    for ( i=0; i < sz; ++i )
        arr->append( UTF8String( authors[i] ) );
    vm->retval( arr );
}


/*#
    @method set_authors GtkAboutDialog
    @brief Sets the strings which are displayed in the authors tab of the secondary credits dialog.
    @param authors array of strings
 */
FALCON_FUNC AboutDialog::set_authors( VMARG )
{
    Item* i_arr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_arr || i_arr->isNil() || !i_arr->isArray() )
        throw_inv_params( "A" );
#endif
    CoreArray* arr = i_arr->asArray();
    AutoCString* cstrings;
    gchar* authors;
    const uint32 num = Gtk::getGCharArray( arr, &authors, &cstrings );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_authors( (GtkAboutDialog*)_obj, (const gchar**)&authors );
    if ( num )
    {
        memFree( authors );
        memFree( cstrings );
    }
}


/*#
    @method get_artists GtkAboutDialog
    @brief Returns the string which are displayed in the artists tab of the secondary credits dialog.
    @return string array containing the artists.
 */
FALCON_FUNC AboutDialog::get_artists( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* const *artists = gtk_about_dialog_get_artists( (GtkAboutDialog*)_obj );
    int sz = 0;
    int i = 0;
    for ( gchar* art = (gchar*) artists[i++]; art; art = (gchar*) artists[i++] )
        ++sz;
    CoreArray* arr = new CoreArray( sz );
    for ( i=0; i < sz; ++i )
        arr->append( UTF8String( artists[i] ) );
    vm->retval( arr );
}


/*#
    @method set_artists GtkAboutDialog
    @brief Sets the strings which are displayed in the artists tab of the secondary credits dialog.
    @param artists array of strings
 */
FALCON_FUNC AboutDialog::set_artists( VMARG )
{
    Item* i_arr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_arr || i_arr->isNil() || !i_arr->isArray() )
        throw_inv_params( "A" );
#endif
    CoreArray* arr = i_arr->asArray();
    AutoCString* cstrings;
    gchar* artists;
    const uint32 num = Gtk::getGCharArray( arr, &artists, &cstrings );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_artists( (GtkAboutDialog*)_obj, (const gchar**)&artists );
    if ( num )
    {
        memFree( artists );
        memFree( cstrings );
    }
}


/*#
    @method get_documenters GtkAboutDialog
    @brief Returns the string which are displayed in the documenters tab of the secondary credits dialog.
    @return string array containing the documenters.
 */
FALCON_FUNC AboutDialog::get_documenters( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* const *documenters = gtk_about_dialog_get_documenters( (GtkAboutDialog*)_obj );
    int sz = 0;
    int i = 0;
    for ( gchar* docu = (gchar*) documenters[i++]; docu; docu = (gchar*) documenters[i++] )
        ++sz;
    CoreArray* arr = new CoreArray( sz );
    for ( i=0; i < sz; ++i )
        arr->append( UTF8String( documenters[i] ) );
    vm->retval( arr );
}


/*#
    @method set_documenters GtkAboutDialog
    @brief Sets the strings which are displayed in the documenters tab of the secondary credits dialog.
    @param documenters array of strings.
 */
FALCON_FUNC AboutDialog::set_documenters( VMARG )
{
    Item* i_arr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_arr || i_arr->isNil() || !i_arr->isArray() )
        throw_inv_params( "A" );
#endif
    CoreArray* arr = i_arr->asArray();
    AutoCString* cstrings;
    gchar* documenters;
    const uint32 num = Gtk::getGCharArray( arr, &documenters, &cstrings );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_documenters( (GtkAboutDialog*)_obj, (const gchar**)&documenters );
    if ( num )
    {
        memFree( documenters );
        memFree( cstrings );
    }
}


/*#
    @method get_translators_credits GtkAboutDialog
    @brief Returns the translator credits string which is displayed in the translators tab of the secondary credits dialog.
    @return The translator credits string.
 */
FALCON_FUNC AboutDialog::get_translator_credits( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* cred = gtk_about_dialog_get_translator_credits( (GtkAboutDialog*)_obj );
    vm->retval( UTF8String( cred ) );
}


/*#
    @method set_translator_credits GtkAboutDialog
    @brief Sets the translator credits string which is displayed in the translators tab of the secondary credits dialog.
    @param translator_credits the translator credits (or nil).
 */
FALCON_FUNC AboutDialog::set_translator_credits( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const char* cred = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_translator_credits( (GtkAboutDialog*)_obj, cred );
}


/*#
    @method get_logo GtkAboutDialog
    @brief Returns the pixbuf displayed as logo in the about dialog.
    @return the pixbuf displayed as logo (or nil).
 */
FALCON_FUNC AboutDialog::get_logo( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GdkPixbuf* buf = gtk_about_dialog_get_logo( (GtkAboutDialog*)_obj );
    if ( buf )
    {
        Gdk::Pixbuf* pix = new Gdk::Pixbuf( vm->findWKI( "GdkPixbuf" )->asClass(), buf );
        vm->retval( pix );
    }
    else
        vm->retnil();
}


/*#
    @method set_logo GtkAboutDialog
    @brief Sets the pixbuf to be displayed as logo in the about dialog.
    @param logo a GdkPixbuf, or nil.

    If it is nil, the default window icon set with gtk_window_set_default_icon() will be used.
 */
FALCON_FUNC AboutDialog::set_logo( VMARG )
{
    Item* i_logo = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_logo || !( i_logo->isNil() ||
        ( i_logo->isObject() && IS_DERIVED( i_logo, GdkPixbuf ) ) ) )
        throw_inv_params( "[GdkPixbuf]" );
#endif
    GdkPixbuf* buf = i_logo->isObject() ? (GdkPixbuf*) COREGOBJECT( i_logo )->getGObject() : NULL;
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_logo( (GtkAboutDialog*)_obj, buf );
}


/*#
    @method get_logo_icon_name GtkAboutDialog
    @brief Returns the icon name displayed as logo in the about dialog.
    @return the icon name displayed as logo.
 */
FALCON_FUNC AboutDialog::get_logo_icon_name( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* nm = gtk_about_dialog_get_logo_icon_name( (GtkAboutDialog*)_obj );
    if ( nm )
        vm->retval( UTF8String( nm ) );
    else
        vm->retnil();
}


/*#
    @method set_logo_icon_name GtkAboutDialog
    @brief Sets the pixbuf to be displayed as logo in the about dialog.
    @param icon_name an icon name, or nil.

    If it is nil, the default window icon set with gtk_window_set_default_icon() will be used.
 */
FALCON_FUNC AboutDialog::set_logo_icon_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    char* nm = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_about_dialog_set_logo_icon_name( (GtkAboutDialog*)_obj, nm );
}


/*#
    @method set_email_hook GtkAboutDialog
    @brief Installs a global function to be called whenever the user activates an email link in an about dialog.
    @param func a function to call when an email link is activated, or nil.
    @param data data to pass to func, or nil.

    The function will get the dialog object as first parameter, the activated link
    as second parameter (string), and user data as third parameter.

    Since 2.18 there exists a default function which uses gtk_show_uri().
    To deactivate it, you can pass NULL for func.
 */
FALCON_FUNC AboutDialog::set_email_hook( VMARG )
{
    Item* i_func = vm->param( 0 );
    Item* i_data = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_func || !( i_func->isNil() || i_func->isCallable() )
        || !i_data )
        throw_inv_params( "[C,X]" );
#endif
    // release anything previously set
    if ( email_hook_func_item )
    {
        gtk_about_dialog_set_email_hook( NULL, NULL, NULL );
        delete email_hook_func_item;
        email_hook_func_item = NULL;
        delete email_hook_data_item;
        email_hook_data_item = NULL;
    }
    // set new func, if any
    if ( !i_func->isNil() )
    {
        email_hook_func_item = new Falcon::GarbageLock( *i_func );
        email_hook_data_item = new Falcon::GarbageLock( *i_data );
        gtk_about_dialog_set_email_hook( &email_hook_func, NULL, NULL );
    }
}

Falcon::GarbageLock*    email_hook_func_item = NULL;
Falcon::GarbageLock*    email_hook_data_item = NULL;

void email_hook_func( GtkAboutDialog* dlg, const gchar* link, gpointer )
{
    assert( email_hook_func_item && email_hook_data_item );

    VMachine* vm = VMachine::getCurrent();

    vm->pushParam( new Gtk::AboutDialog( vm->findWKI( "GtkAboutDialog")->asClass(), dlg ) );
    vm->pushParam( UTF8String( link ) );
    vm->pushParam( email_hook_data_item->item() );
    vm->callItem( email_hook_func_item->item(), 3 );
}


/*#
    @method set_url_hook GtkAboutDialog
    @brief Installs a global function to be called whenever the user activates a URL link in an about dialog.
    @param func a function to call when a URL link is activated, or nil.
    @param data data to pass to func, or nil.

    The function will get the dialog object as first parameter, the activated link
    as second parameter (string), and user data as third parameter.

    Since 2.18 there exists a default function which uses gtk_show_uri().
    To deactivate it, you can pass NULL for func.
 */
FALCON_FUNC AboutDialog::set_url_hook( VMARG )
{
    Item* i_func = vm->param( 0 );
    Item* i_data = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_func || !( i_func->isNil() || i_func->isCallable() )
        || !i_data )
        throw_inv_params( "[C,X]" );
#endif
    // release anything previously set
    if ( url_hook_func_item )
    {
        gtk_about_dialog_set_url_hook( NULL, NULL, NULL );
        delete url_hook_func_item;
        url_hook_func_item = NULL;
        delete url_hook_data_item;
        url_hook_data_item = NULL;
    }
    // set new func, if any
    if ( !i_func->isNil() )
    {
        url_hook_func_item = new Falcon::GarbageLock( *i_func );
        url_hook_data_item = new Falcon::GarbageLock( *i_data );
        gtk_about_dialog_set_url_hook( &url_hook_func, NULL, NULL );
    }
}

Falcon::GarbageLock*    url_hook_func_item = NULL;
Falcon::GarbageLock*    url_hook_data_item = NULL;

void url_hook_func( GtkAboutDialog* dlg, const gchar* link, gpointer )
{
    assert( url_hook_func_item && url_hook_data_item );

    VMachine* vm = VMachine::getCurrent();

    vm->pushParam( new Gtk::AboutDialog( vm->findWKI( "GtkAboutDialog")->asClass(), dlg ) );
    vm->pushParam( UTF8String( link ) );
    vm->pushParam( url_hook_data_item->item() );
    vm->callItem( url_hook_func_item->item(), 3 );
}


//FALCON_FUNC AboutDialog::show_about_dialog( VMARG );


} // Gtk
} // Falcon

#endif // GTK_MINOR_VERSION >= 6
